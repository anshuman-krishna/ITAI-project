from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

# the wire format is camelCase to match packages/shared and idiomatic frontend code, while
# python keeps snake_case fields. populate_by_name lets us build models from the snake_case
# ml-core dataclasses; fastapi serializes responses by alias (camelCase).


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
