from expenvelope.json_serializer import SavesToJSON


class Seed(SavesToJSON):

    def __init__(self, foo):
        self.foo = foo

    def _to_dict(self) -> dict:
        return {"foo": self.foo}

    @classmethod
    def _from_dict(cls, json_dict):
        return cls(**json_dict)


class Fruit(SavesToJSON):

    def __init__(self, seeds):
        self.seeds = seeds

    def _to_dict(self) -> dict:
        return {"seeds": self.seeds}

    @classmethod
    def _from_dict(cls, json_dict):
        return cls(**json_dict)


fruit_with_seeds = Fruit([Seed(6), Seed(-80), Fruit([Seed(-1), Seed(3)]), Seed("hello")])
print(fruit_with_seeds)
json_version = fruit_with_seeds.json_dumps()
print(json_version)
print(Fruit.json_loads(json_version))
