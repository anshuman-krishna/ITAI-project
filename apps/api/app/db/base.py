from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    # all orm models inherit from this. models are registered by importing them
    # in app.models so alembic and metadata can see them.
    pass
