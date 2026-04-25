# Legislative Module Database

class LegislativeModule:
	
    def __init__(self):
        # Datastructure
        self.laws = {
            "Law1":{
                "NL": "Don't grab mickey",
                "Symbolic":{
                    "Object": "mickey",
                    "Predicate": "Grasp",
                    "Consequence": "Release",
                    }
                }
            }

    def get_all_laws(self) -> list[dict]:
        return list(self.laws.values())

    def get_num_laws(self) -> int:
        return len(self.laws.keys())

    def get_illegal_objects(self) -> list[str]:
        return [law["Symbolic"]["Object"] for law in self.laws.values()]

    def get_law_for_object(self, object_name: str | list) -> list[dict]:
        if isinstance(object_name, str):
            object_name = [object_name]
        return [law for law in self.laws.values() if law["Symbolic"]["Object"] in object_name]

    def get_predicate(self, object_name: str) -> bool:
        law = self.get_law_for_object(object_name)
        return law.predicate if law else None

    def get_consequence(self, object_name: str) -> str | None:
        law = self.get_law_for_object(object_name)
        return law.consequence if law else None
