import json
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker

from learning_engine.config import DATABASE_URI

Base = declarative_base()

class FeedbackDB(Base):
    __tablename__ = 'feedback_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String, unique=True, index=True)
    file_hash = Column(String, index=True)
    media_type = Column(String)
    original_verdict = Column(String)
    corrected_verdict = Column(String)
    module_scores_json = Column(Text)
    flags_json = Column(Text)
    timestamp = Column(Float)
    metadata_json = Column(Text)

    @property
    def module_scores(self):
        return json.loads(self.module_scores_json) if self.module_scores_json else {}

    @module_scores.setter
    def module_scores(self, value):
        self.module_scores_json = json.dumps(value)

    @property
    def flags(self):
        return json.loads(self.flags_json) if self.flags_json else []

    @flags.setter
    def flags(self, value):
        self.flags_json = json.dumps(value)

    @property
    def metadata_dict(self):
        return json.loads(self.metadata_json) if self.metadata_json else {}

    @metadata_dict.setter
    def metadata_dict(self, value):
        self.metadata_json = json.dumps(value)


class PatternDB(Base):
    __tablename__ = 'known_patterns'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vector_json = Column(Text)
    keys_json = Column(Text)
    manipulation_type = Column(String)
    flags_json = Column(Text)
    timestamp = Column(Float)

    @property
    def vector(self):
        return json.loads(self.vector_json) if self.vector_json else []

    @vector.setter
    def vector(self, value):
        self.vector_json = json.dumps(value)

    @property
    def keys(self):
        return json.loads(self.keys_json) if self.keys_json else []

    @keys.setter
    def keys(self, value):
        self.keys_json = json.dumps(value)

    @property
    def flags(self):
        return json.loads(self.flags_json) if self.flags_json else []

    @flags.setter
    def flags(self, value):
        self.flags_json = json.dumps(value)


class ThresholdStatsDB(Base):
    __tablename__ = 'threshold_stats'
    
    module_name = Column(String, primary_key=True)
    auth_scores_json = Column(Text)
    fake_scores_json = Column(Text)
    adjustment = Column(Float, default=0.0)

    @property
    def auth_scores(self):
        return json.loads(self.auth_scores_json) if self.auth_scores_json else []

    @auth_scores.setter
    def auth_scores(self, value):
        self.auth_scores_json = json.dumps(value)

    @property
    def fake_scores(self):
        return json.loads(self.fake_scores_json) if self.fake_scores_json else []

    @fake_scores.setter
    def fake_scores(self, value):
        self.fake_scores_json = json.dumps(value)


# Global Engine and Session Factory
engine = create_engine(DATABASE_URI, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

# Initialize database schema upon import
init_db()
