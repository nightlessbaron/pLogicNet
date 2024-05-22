import os
import json


def write_to_file(path, entities, relations):
    with open(os.path.join(path, "entities.dict"), "w") as f:
        for entity in entities:
            f.write(str(entities[entity]) + '\t' + entity + '\n')
    with open(os.path.join(path, "relations.dict"), "w") as f:
        for relation in relations:
            f.write(str(relations[relation]) + '\t' + relation + '\n')


def generate_entities_relations(path):
    # Load the data from vocabulary.json
    with open(os.path.join(path, 'vocabulary.json'), 'r') as f:
        data = json.load(f)
    entities = data['objects']
    relations = data['predicates']
    write_to_file(path, entities, relations)


def convert_data_to_triples(path, split="train"):
    # Load the data from split.nl file
    with open(os.path.join(path, split + '.nl'), 'r') as f:
        nl_data = f.readlines()
    # data in each line is like this relation(object1, object2).
    # I want to convert it to object1 relation object2
    txt_data = []
    for line in nl_data:
        relation, objects = line.split('(')
        object1, object2 = objects.split(',')
        object2 = object2[:-3]
        txt_data.append(object1 + '\t' + relation + '\t' + object2 + '\n')

    with open(os.path.join(path, split + '.txt'), 'w') as f:
        for line in txt_data:
            f.write(line)


def main():
    # please change the data path to the path of the data folder
    path = "data/nations"
    generate_entities_relations(path)
    convert_data_to_triples(path, "train")
    convert_data_to_triples(path, "val")
    convert_data_to_triples(path, "test")


if __name__ == "__main__":
    main()
