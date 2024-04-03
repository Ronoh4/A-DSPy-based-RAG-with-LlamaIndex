# Import modules
import os
import sys
import dspy
import pkg_resources
from dspy import Signature, InputField, OutputField, Module, Predict, Prediction
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match, answer_passage_match
from dspy import Example

# Set environmental variables
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx7M"
openai_api_key = os.environ["OPENAI_API_KEY"]

# Define path to project
repo_path = 'C:\\Users\\user\\Documents\\Jan 2024\\Projects\\RAGs\\New\\DSPy\\DSPyRAG'

# Add the project path to your system path
if repo_path not in sys.path:
    sys.path.append(repo_path)

# Set up the cache for this script
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(repo_path, 'cache')

# Check if dspy-ai is installed
if not "dspy-ai" in {pkg.key for pkg in pkg_resources.working_set}:
    print("Please install dspy-ai and openai using pip")

# Configure LM
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo)

# Parse file
parser = LlamaParse(
    api_key="llx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxGh7",
        result_type="text",
        language="en",
        varbose=True
    )

# Create documents and index
documents = parser.load_data("C:\\Users\\user\\Documents\\Jan 2024\\Projects\\RAGs\\Files\\PhilDataset.pdf")
print("Documents created")
index = VectorStoreIndex.from_documents(documents)

index.set_index_id("vector_index")
index.storage_context.persist("./storage")

storage_context = StorageContext.from_defaults(persist_dir="storage")

# Create query engine as index
index = load_index_from_storage(storage_context, index_id="vector_index")
query_engine = index.as_query_engine(response_mode="tree_summarize")

# Create signature
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Often between 5 and 10 words")
    print("Class 1 created")

# Define modules
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.query_engine = query_engine
        self.generate_answer = Predict(GenerateAnswer)
        print("Class 2 created")

    def forward(self, question):
        response = self.query_engine.query(question)
        context = response.response
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
custom_rag = RAG(query_engine)

question = "What did Phil wanted to become when he grew up?"
pred = custom_rag(question)
print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")

# Create validation logic 
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = answer_exact_match(example, pred)
    answer_PM = answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Define examples with the necessary fields
train_example1 = Example(question="What did young Philemon wanted to become when he grew up?", answer="Engineer")
train_example2 = Example(question="What did Philemon realize his curiosity was pushing him towards as he grew older?", answer="Sciences")
train_example3 = Example(question="How many years after graduation did Philemon spent working in the academic writing industry?", answer="Eight")
train_example4 = Example(question="Which is one of the subjects that Philemon handled in academic writing assignments?", answer="Nursing")
train_example5 = Example(question="What made the global academic system to go into hibernation?", answer="Covid")
train_example6 = Example(question="Which year did the usual peak season failed to materialize?", answer="2021")
train_example7 = Example(question="When was the ranking systems introduced to deny all other writers the chance to see available orders?", answer="2023")
train_example8 = Example(question="In 2024, how many orders had Philemon completed until February 15?", answer="4")
train_example9 = Example(question="What was the main reason Philemon wanted to branch into other high-demand fields?", answer="Income")
train_example10 = Example(question="What did Philemon eventually venture into in his undergraduate studies?", answer="Chemistry")

# Tell DSPy that the 'question' field is the input
trainset = [
    train_example1.with_inputs('question'),
    train_example2.with_inputs('question'),
    train_example3.with_inputs('question'),
    train_example4.with_inputs('question'),
    train_example5.with_inputs('question'),
    train_example6.with_inputs('question'),
    train_example7.with_inputs('question'),
    train_example8.with_inputs('question'),
    train_example9.with_inputs('question'),
    train_example10.with_inputs('question'),
]

print("Trainset created")

# Set up teleprompter
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

compiled_rag = teleprompter.compile(custom_rag, trainset=trainset)

# Use compiled_rag to answer questions about your PDF!
question = "When did the rationing of orders took a policy direction?"
pred = compiled_rag(question)
print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
print("Retrieved Contexts:")
for context in pred.context:
    full_context = ''.join(context)
    print(full_context)


#Output
#Started parsing the file under job_id 65bd7202-7285-44d3-8f02-7a1a115a4367
#Documents created

#Question: What did Phil wanted to become when he grew up?
#Predicted Answer: An engineer

#100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [03:00<00:00, 18.05s/it]
#Bootstrapped 1 full traces after 10 examples in round 0.

#Question: When did the rationing of orders took a policy direction?
#Predicted Answer: 2023
#Retrieved Contexts:
#The rationing of orders took a policy direction in 2023.