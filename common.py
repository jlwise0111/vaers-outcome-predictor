from sqlalchemy import create_engine


######
## Database connection
######
engine = create_engine('postgresql://postgres:psw@localhost:5432/postgres')

MODEL_PATH = "trained_model.joblib"