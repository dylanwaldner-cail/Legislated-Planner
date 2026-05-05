

class LegislativeModule:
    def __init__(self):
        ''' Currently hardcoded for point maze'''
        self.laws = {"Law1": 
                            {"type": "illegal_region", "illegal_region": [[0, 1], [1, 2]]},
                    }

    def _get_laws_by_index(self, indices: list):
        laws = dict()
        for idx in indices:
            key = f'Law{idx}'
            try:
                laws[key] = self.laws[key]
            except ValueError:
                print(f'{idx} not a valid index')

        return laws

    def _get_laws_by_types(self, law_types: list):
        laws = dict()
        for key, value in self.laws.items():
            if value["type"] in law_types:
                laws[key] = self.laws[key]

        return laws


    def add_law(self):
        ''' To do '''
            
