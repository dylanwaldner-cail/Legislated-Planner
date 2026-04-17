# Legislative Module Database

class LegislativeModule:
	
    def __init__(self):
        # Datastructure
        self.laws = {
            "Law1":{
                "NL": "Don't grab the bowl",
                    "PDDL":{
                        "Object": "akita_black_bowl_1",
                        "Predicate": "Grasp",
                        "Consequence": "Release",
                    }
                }
            }

    def get_laws(self) -> dict:
        return self.laws

    def get_num_laws(self) -> int:
        return len(self.laws.keys())

    def get_illegal_objects(self) -> list[str]:
        return [law["PDDL"]["Object"] for law in self.laws.values()]

    def get_law_for_object(self, object_name: str) -> dict | None:
        for law in self.laws.values():
            if law["PDDL"]["Object"] == object_name:
                return law
        return None

    def get_predicate(self, object_name: str) -> bool:
        law = self.get_law_for_object(object_name)
        return law.predicate if law else None

    def get_consequence(self, object_name: str) -> str | None:
        law = self.get_law_for_object(object_name)
        return law.consequence if law else None
