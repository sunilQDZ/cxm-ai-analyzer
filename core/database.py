import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import config

_user = config.MYSQL_USER
_password = urllib.parse.quote_plus(config.MYSQL_PASSWORD)
_host = config.MYSQL_HOST
_port = config.MYSQL_PORT
_db = config.MYSQL_DB

DATABASE_URL = f"mysql+pymysql://{_user}:{_password}@{_host}:{_port}/{_db}?charset=utf8mb4"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
