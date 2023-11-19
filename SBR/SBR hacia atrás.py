# Definimos una clase para las reglas
class Rule:
    def __init__(self, id, conditions, result):
        self.id = id
        self.conditions = conditions
        self.result = result

# Definimos una clase para el motor de inferencia
class InferenceEngine:
    def __init__(self, rules):
        self.rules = rules
        self.facts = set()

    # Definimos un método para el encadenamiento hacia atrás
    def backward_chaining(self, goal):
        if goal in self.facts:
            return True
        for rule in sorted(self.rules, key=lambda x: x.id):
            if rule.result == goal:
                if all(self.backward_chaining(condition) for condition in rule.conditions):
                    self.facts.add(rule.result)
                    return True
        return False

# Definimos las reglas como una lista de objetos de la clase Rule
rules = [
    Rule(1, ['b', 'd', 'e'], 'f'),
    Rule(2, ['d', 'g'], 'a'),
    Rule(3, ['c', 'f'], 'a'),
    Rule(4, ['b'], 'x'),
    Rule(5, ['d'], 'e'),
    Rule(6, ['a', 'x'], 'h'),
    Rule(7, ['c'], 'd'),
    Rule(8, ['x', 'c'], 'a'),
    Rule(9, ['x', 'b'], 'd')
]

# Creamos un objeto del motor de inferencia con las reglas definidas
engine = InferenceEngine(rules)
engine.facts = {'b', 'c'}  # Conjunto inicial de hechos

goal = 'h'  # Meta a alcanzar

# Si podemos alcanzar la meta con el encadenamiento hacia atrás
if engine.backward_chaining(goal):
    print(f"La meta '{goal}' se puede alcanzar.")  # Imprimimos que la meta se puede alcanzar
else:
    print(f"La meta '{goal}' no se puede alcanzar.")  # Imprimimos que la meta no se puede alcanzar

print("Hechos inferidos:", engine.facts)  # Imprimimos los hechos inferidos
