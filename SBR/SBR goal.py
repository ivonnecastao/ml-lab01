# Definimos una clase para las reglas
class Rule:
    # El método de inicialización toma un id, condiciones y resultado
    def __init__(self, id, conditions, result):
        self.id = id  # El id de la regla
        self.conditions = conditions  # Las condiciones de la regla
        self.result = result  # El resultado de la regla

# Definimos una clase para el motor de inferencia
class InferenceEngine:
    # El método de inicialización toma un conjunto de reglas
    def __init__(self, rules):
        self.rules = rules  # Las reglas del motor de inferencia
        self.facts = set()  # Los hechos conocidos por el motor de inferencia

    # Definimos un método para el encadenamiento hacia adelante
    def forward_chaining(self, goal):
        used_rules = set()  # Conjunto para almacenar las reglas utilizadas
        while True:  # Mientras haya reglas por procesar
            new_facts = set()  # Conjunto para almacenar los nuevos hechos inferidos
            for rule in sorted(self.rules, key=lambda x: x.id):  # Para cada regla en las reglas ordenadas por id
                if rule.id not in used_rules and all(condition in self.facts for condition in rule.conditions):  # Si la regla no ha sido utilizada y todas las condiciones se cumplen
                    used_rules.add(rule.id)  # Añadimos la regla a las reglas utilizadas
                    new_facts.add(rule.result)  # Añadimos el resultado a los nuevos hechos
                    if goal in new_facts:  # Si hemos alcanzado la meta
                        return True  # Retornamos verdadero
            if not new_facts:  # Si no hemos inferido nuevos hechos
                break  # Salimos del bucle while
            self.facts |= new_facts  # Añadimos los nuevos hechos a los hechos conocidos
        return False  # Si no hemos alcanzado la meta, retornamos falso

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
engine.facts = {'b', 'c'}  # Definimos el conjunto inicial de hechos

goal = 'h'  # Definimos la meta a alcanzar

# Si podemos alcanzar la meta con el encadenamiento hacia adelante
if engine.forward_chaining(goal):
    print(f"La meta '{goal}' se puede alcanzar.")  # Imprimimos que la meta se puede alcanzar
    print(engine.facts)  # Imprimimos los hechos inferidos
else:
    print(f"La meta '{goal}' no se puede alcanzar.")  # Imprimimos que la meta no se puede alcanzar
    print(engine.facts)  # Imprimimos los hechos inferidos


"""
El código que has pedido hace lo siguiente:

Define una clase Rule para representar las reglas del sistema basado en reglas. Cada regla tiene un identificador, una lista de condiciones y un resultado.
Define una clase InferenceEngine para implementar el motor de inferencia que utiliza la estrategia de encadenamiento hacia adelante y el sistema básico de inferencia modus ponens. Cada instancia de esta clase tiene una lista de reglas y un conjunto de hechos.
Define el método forward_chaining que recibe una meta como parámetro y devuelve un valor booleano que indica si la meta se puede alcanzar o no con las reglas y hechos dados. Este método aplica las reglas en orden ascendente de su identificador, y solo utiliza cada regla una vez. Si se encuentra un nuevo hecho que coincide con la meta, el método termina y devuelve True. Si no se encuentran nuevos hechos, el método termina y devuelve False.
Crea una lista de reglas con los datos que has proporcionado. Cada regla se crea con un objeto de la clase Rule.
Crea un objeto de la clase InferenceEngine con la lista de reglas. Asigna al atributo facts el conjunto inicial de hechos que has proporcionado.
Asigna a la variable goal la meta que quieres alcanzar. Llama al método forward_chaining con esta variable como argumento. Imprime un mensaje que indica si la meta se puede alcanzar o no. Puedes cambiar la meta según sea necesario.

El motor de inferencia implementado en el código anterior es un sistema basado en reglas que utiliza la estrategia de encadenamiento hacia adelante y el sistema básico de inferencia modus ponens. Aquí está una descripción detallada de cómo funciona:

Clase Rule: Cada regla en el sistema se representa como una instancia de la clase Rule. Cada regla tiene un identificador único (id), una lista de condiciones (conditions) que deben cumplirse para que la regla se aplique, y un resultado (result) que se agrega a los hechos conocidos si la regla se aplica.

Clase InferenceEngine: Esta es la clase principal que implementa el motor de inferencia. Tiene una lista de rules (reglas) y un conjunto de facts (hechos) conocidos. Cuando se crea una instancia de esta clase, se le proporciona una lista de reglas.

Método forward_chaining: Este método implementa la estrategia de encadenamiento hacia adelante. Comienza con los hechos conocidos e intenta aplicar las reglas para inferir nuevos hechos. Las reglas se aplican en orden ascendente de su identificador, y cada regla solo se puede utilizar una vez. Si se encuentra un nuevo hecho que coincide con la meta, el método termina y devuelve True. Si no se encuentran nuevos hechos, el método termina y devuelve False.

Ejecución del motor de inferencia: Se crea una lista de reglas y un conjunto inicial de hechos. Luego, se crea una instancia del motor de inferencia con estas reglas y hechos. Finalmente, se llama al método forward_chaining con la meta deseada.

En resumen, este motor de inferencia utiliza un sistema basado en reglas y la estrategia de encadenamiento hacia adelante para alcanzar una meta dada a partir de un conjunto inicial de hechos.

"""