import os
import shutil
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from spacy.training.example import Example
from spacy.util import minibatch
import random
from training_data import data

sentences = [
    "make a folder xyz", "create new folder project", "add a file notes.txt",
    "create file test.py", "new file report.txt", "create a new file called data.csv",
    "make file notes.md","generate file summary.docx","add file script.py",
    "new folder images", "create a new folder called docs", "make directory backups",
    "add folder music", "generate folder videos","create folder name happy and jump",
    "make these files new.txt and old.txt","new folders willy and wonka neede", "need the files afhsan.py and kaushal.py here",
    "can u make a folder called images, static, templates", "can u make files called acer.py ros.cpp and node.sh",

    "make fil.txt", "create fil.txt", "add fil.txt",
    "generate fil.txt", "new fil.txt","make file fil.txt",
    "create file fil.txt", "add file fil.txt", "generate file fil.txt", 
    "new file fil.txt",

    "delete folder old_project", "remove file trash.py", "erase folder backup",
    "delete file named creative stuff", "eliminate folder named delete these","delete folder named create jobs",
    "delete file named stuff","remove directory temp", "delete the file called report.docx",
    "erase directory logs", "destroy a file output.txt", "eliminate folder test_data",
    "remove report.txt", "delete the file data.csv", "erase file notes.md",
    "eliminate script.py", "trash file summary.docx","remove images",
    "delete the folder docs", "erase folder backups","eliminate directory music",
    "trash folder videos","erase happy and jump folder","remove flash.py home.html and home.css",
    "remove all these aayush and harshvardhan","eliminate folder final_older and final_oldest","wipe these files latest_edit.py term_project.py",
    "discard new folders i made"," discard all files in the directory python",

    "delete fil.txt", "erase fil.txt", "remove fil.txt",
    "wipe fil.txt", "trash fil.txt","wipe out file fil.txt",
    "destroyfile fil.txt", "eliminate file fil.txt", "discard file fil.txt", 
]

labels = [
    "create", "create", "create",
    "create", "create", "create",
    "create", "create", "create",
    "create", "create", "create",
    "create", "create", "create",
    "create", "create", "create",
    "create", "create",

    "create", "create", "create",
    "create", "create", "create",
    "create", "create", "create",
    "create", 

    "delete", "delete", "delete",
    "delete", "delete", "delete",
    "delete", "delete", "delete",
    "delete", "delete", "delete",
    "delete", "delete", "delete",
    "delete", "delete", "delete",
    "delete", "delete", "delete",
    "delete", "delete", "delete",
    "delete", "delete", "delete",
    "delete", "delete",

    "delete", "delete", "delete",
    "delete", "delete", "delete",
    "delete", "delete", "delete",
]


intent_model = make_pipeline(
    CountVectorizer(ngram_range=(1, 3), analyzer='char_wb'),
    LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
)
intent_model.fit(sentences, labels)



def get_fixed_training_data():

    TRAIN_DATA = []

    for sentence, things in data:
        entities = []
        for thing, label in things:
            start = sentence.find(thing)
            if start != -1:
                end = start + len(thing)
                entities.append((start, end, label.upper()))
        TRAIN_DATA.append((sentence, {"entities": entities}))

    return TRAIN_DATA

TRAIN_DATA = get_fixed_training_data()

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

ner.add_label("FILE")
ner.add_label("FOLDER")

optimizer = nlp.begin_training()
for itn in range(150):
    random.shuffle(TRAIN_DATA)
    losses = {}
    batches = minibatch(TRAIN_DATA, size=2)
    for batch in batches:
        examples = []
        for text, annots in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annots)
            examples.append(example)
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Iteration {itn+1}, Losses: {losses}")


def extract_name(command):
    name_model = nlp(command)
    file_name=[ent.text for ent in name_model.ents if ent.label_ == "FILE"]
    folder_name=[ent.text for ent in name_model.ents if ent.label_ == "FOLDER"]

    names={
        "file" : [name for name in file_name],
        "folder" : [name for name in folder_name]
    }

    return names

def predict_intent(command):
    return intent_model.predict([command])[0]

def to_do_commands(intent, names):
    file_names=[name for name in names["file"]]
    folder_names=[name for name in names["folder"]]

    try:
        if intent == "create":
            for name in folder_names:
                os.makedirs(name, exist_ok=True)
            for name in file_names:
                open(name, 'w').close()
        elif intent == "delete":
            for name in folder_names:
                if os.path.isdir(name):
                    shutil.rmtree(name)
                else:
                  return f"Folder '{name}' does not exist."
            for name in file_names:
                if os.path.isfile(name):
                    os.remove(name)
                else:
                    return f"File '{name}' does not exist."
    except Exception as e:
            return f"Error: {str(e)}"

#main
    
if __name__ == "__main__":
    print("Welcome to AI Terminal, enter your commands to make or delete folders, type quit to exit")

    while(True):
        # os.system('cls' if os.name == 'nt' else 'clear')
        print("$ ",end="")
        command=input()

        if command.lower() =="quit":
            break
        names = extract_name(command)
        intent = predict_intent(command)
        result = to_do_commands(intent, names)
