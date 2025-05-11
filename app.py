import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

from common import MODEL_PATH, engine


@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)

def filters_to_sql(sex, age_min, age_max, year_min, year_max, outcomes):
    sql = "\"YEAR\" between " + str(year_min) + " and " + str(year_max) +\
          " AND  \"AGE_YRS\" between " +\
          str(age_min) + " and " + str(age_max)
    if sex != 'All':
        sql += " AND \"SEX\" = '" + str(sex) + "'"
    if outcomes != 'All':
        sql += " AND \"OUTCOME\" = '" + str(outcomes) + "'"

    return sql


with st.sidebar:
    st.header('Filters for the Data Dashboard')
    selected_sex = st.selectbox('Filter by Sex', options=['All', 'F', 'M', 'U'])
    outcomes = st.selectbox('Filter by Outcome', options=['All', 'Death', 'Hospitalization', 'ER Visit'])
    (age_min, age_max) = st.slider('Filter by Age', min_value=0, max_value=112, value=(0, 112))
    (year_min, year_max) = st.slider("Select Year", min_value=1990, max_value=2025, value=(1990, 2025))

filter_str = filters_to_sql(selected_sex, age_min, age_max, year_min, year_max, outcomes)


tab1, tab2, tab3 = st.tabs(["Dashboard", "Prediction", "Data Explorer"])

with tab1:
    st.title('VAERS Data Dashboard')

    outcome_cnts = pd.read_sql_query(f"""
    select "OUTCOME", count(*) as cnt
    from vaers_data
    where {filter_str}
    group by "OUTCOME"
    order by cnt desc
    """,con=engine)

    st.subheader("Distribution of Outcomes")
    fig = px.pie(
        outcome_cnts,
        values = "cnt",
        names = "OUTCOME",
        title = 'Outcome Distribution',
        color_discrete_sequence = px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)

    top_by_vaccine_type = pd.read_sql_query(f"""
    select "VAX_TYPE", count(*) as cnt
    from vaers_data
    where {filter_str}
    group by "VAX_TYPE"
    order by cnt desc
    """, con=engine).sort_values(by='cnt', ascending=False)
    st.subheader("Top Vaccine Types Reported")
    st.bar_chart(top_by_vaccine_type, x="VAX_TYPE", y="cnt")

    adverse_reactions = pd.read_sql_query(f"""
    select
        "YEAR",
        "VAX_TYPE",
        count(*) as report_count
    from vaers_data
    where {filter_str}
    group by "YEAR", "VAX_TYPE"
    order by "YEAR" asc, report_count desc;
    """, con=engine).sort_values(by='report_count', ascending=False)
    st.subheader("Adverse Reactions Reported Over Time")
    st.bar_chart(adverse_reactions, x="YEAR", y="report_count")


    st.subheader("Top 10 Vaccines/Reactions")
    df = pd.read_sql_query(f"""
    select
        "VAX_TYPE" || ' - ' || "SYMPTOM1" as label,
        count(*) as report_count
    from vaers_data
    where {filter_str}
    group by "SYMPTOM1", "VAX_TYPE"
    order by report_count desc
    limit 10;
    """, con=engine).sort_values(by='report_count', ascending=False)
    st.bar_chart(df, x="label", y="report_count")


    st.subheader("Top 10 Overall Reactions")

    df = pd.read_sql_query(f"""
    select
        "SYMPTOM1" as symptom,
        count(*) as report_count
    from vaers_data
    where {filter_str} and "SYMPTOM1" is not null
    group by "SYMPTOM1"
    order by report_count desc
    limit 10;
    """, con=engine).sort_values(by='report_count', ascending=False)
    st.bar_chart(df, x="symptom", y="report_count")

    age_dist = pd.read_sql_query(f"""
        SELECT
            CONCAT(FLOOR("AGE_YRS" / 10) * 10, '-', FLOOR("AGE_YRS" / 10) * 10 + 9) AS age_group,
            COUNT(*) AS report_count
        FROM vaers_data
        WHERE "OUTCOME" != 'No hospitalization, ER visit, or death' AND {filter_str}
        GROUP BY age_group
        ORDER BY MIN("AGE_YRS") ASC;
    """, con=engine)

    st.subheader("Age Distribution of Serious Outcomes")
    st.bar_chart(age_dist, x="age_group", y="report_count")

with tab2:
    st.header("Predicting Serious Outcome Based on Selections")

    model = load_model()

    st.markdown("### Input Features for Prediction")

    df_vax = pd.read_sql_query(f"""
        select distinct "VAX_TYPE"
        from vaers_data
        where "VAX_TYPE" is not null
    
    """, con=engine)

    vax_options = sorted(df_vax["VAX_TYPE"].dropna().unique())

    input_sex = st.selectbox("Sex", options=["F", "M", "U"], key="input_sex")
    input_age = st.slider("Age", 0, 112, 30, key="input_age")
    input_vax_type = st.selectbox("Vaccine Type", options=vax_options, key="input_vax")
    input_dose = st.selectbox("Dose Number", options=range(1, 6), key="input_dose")

    input_df = pd.DataFrame([{
        'SEX': input_sex,
        'AGE_YRS': int(input_age),
        'VAX_TYPE': input_vax_type,
        'VAX_DOSE_SERIES': int(input_dose)
    }])


    pred_tmp = model.predict(input_df)
    prediction = pred_tmp[0]
    probabilities = model.predict_proba(input_df)

    st.markdown("### Prediction Result")
    st.write(f"**Predicted Outcome:** {prediction}")

    st.write(f"**Confidence:** {probabilities.max() * 100:.2f}%")
    for i in range(0, 4):
        st.write(f"**Confidence for {model.classes_[i]}:** {probabilities[0, i] * 100:.2f}%")

with tab3:
    tbl = pd.read_sql_query("select * from vaers_data where " + filter_str + " limit 100;", con=engine)

    st.dataframe(tbl, hide_index=True)