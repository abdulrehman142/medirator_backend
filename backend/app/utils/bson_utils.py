from bson import ObjectId


def oid(value: str) -> ObjectId:
    return ObjectId(value)


def _normalize_bson(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalize_bson(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_bson(item) for item in value]
    return value


def to_str_id(document: dict) -> dict:
    if not document:
        return document
    data = _normalize_bson(dict(document))
    if "_id" in data:
        data["id"] = data.pop("_id")
    return data
