import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# Initialize an empty task list
tasks = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks from a CSV file (if any)
try:
    tasks = pd.read_csv('C:/Users/dinu3/OneDrive/Desktop/basic/tasks.csv')
except FileNotFoundError:
    pass

# Function to save tasks to a CSV file
def save_tasks():
    tasks.to_csv('C:/Users/dinu3/OneDrive/Desktop/basic/tasks.csv', index=False)

# Train the task priority classifier
vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)

# Only train the model if there are existing tasks
if not tasks.empty:
    model.fit(tasks['description'], tasks['priority'])

# Function to add a task to the list
def add_task(description, priority=None):
    global tasks  # Declare tasks as a global variable
    if priority is None and not tasks.empty:
        priority = model.predict([description])[0]  # Predict priority if not provided
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()
    # Retrain the model with the updated tasks
    if not tasks.empty:
        model.fit(tasks['description'], tasks['priority'])

# Function to remove a task by description
def remove_task(description):
    global tasks  # Declare tasks as a global variable
    tasks = tasks[tasks['description'] != description]
    save_tasks()
    # Retrain the model with the updated tasks
    if not tasks.empty:
        model.fit(tasks['description'], tasks['priority'])

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

# Function to recommend a task based on machine learning
def recommend_task():
    if not tasks.empty:
        tasks['predicted_priority'] = model.predict(tasks['description'])
        
        # Order of priorities to check
        priorities = ['High', 'Medium', 'Low']
        
        for priority in priorities:
            priority_tasks = tasks[tasks['predicted_priority'] == priority]
            if not priority_tasks.empty:
                random_task = random.choice(priority_tasks['description'].tolist())
                print(f"Recommended task: {random_task} - Priority: {priority}")
                return
        
        print("No tasks available for recommendation.")
    else:
        print("No tasks available for recommendations.")

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High) or leave blank for auto: ").capitalize() or None
        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")
