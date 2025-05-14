import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import tiktoken
import re
from decouple import config
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# Image classification model class
class DCNN(nn.Module):
    def __init__(self, num_classes):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        if self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 512).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PlantDiseaseRAG:
    def __init__(self, openai_api_key, persist_directory="plant_disease_db"):
        """Initialize the RAG system with OpenAI API key"""
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(
                api_key=openai_api_key,
                model="text-embedding-3-small"
            )
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(
    """
    You are an expert agricultural consultant specializing in plant diseases.
    You have been given information about a plant disease detected through image analysis.

    DETECTED CONDITION: {detected_disease}

    RELEVANT INFORMATION:
    {context}

    Based on the detected condition and the information provided, give concise and practical advice to the farmer.
    Include:
    1. A brief explanation of the disease/condition
    2. Immediate actions to take
    3. Long-term management strategies
    4. Prevention tips for future crops

    Be specific and actionable. If the plant is healthy, focus on maintenance and prevention.
    Respond with a very short summary containing only the most essential recommendations. Do not begin with the word 'Summary:'.
    """)

        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.openai_api_key,
            temperature=0.2,
            max_tokens=600
        )
        
        # Create the chain
        self.chain = (
            {"context": self.retriever, "detected_disease": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def get_advice(self, disease_class):
        """Get advice for a specific disease class"""
        return self.chain.invoke(disease_class)

def diagnose_plant_disease(image_path, model_path, class_names_path):
    """
    Diagnose plant disease from an image
    
    Args:
        image_path: Path to the plant image
        model_path: Path to the trained PyTorch model
        class_names_path: Path to the class names JSON file
        
    Returns:
        Predicted disease class
    """
    # Load class names
    with open(class_names_path) as f:
        class_names = json.load(f)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    
    return predicted_class

def main():
    """Main function to set up and run the system"""
    # Set up API key
    OPENAI_API_KEY = config("AGRANOVA")
    
    # Prepare the dataset and create vector store (run once)

    
    # Initialize the RAG system
    rag_system = PlantDiseaseRAG(openai_api_key=OPENAI_API_KEY)
    
    # Example usage with an image
    image_path = "C:/Users/anis/OneDrive/Desktop/PlantleafCnn/0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG" 
    model_path = "best_dcnn_model.pth"  # Replace with your model path
    class_names_path = "class_names.json"  # Replace with your class names file
    
    # Diagnose the plant disease
    predicted_class = diagnose_plant_disease(image_path, model_path, class_names_path)
    print(f"Predicted Class: {predicted_class}")
    
    # Get advice using RAG
    advice = rag_system.get_advice(predicted_class)
    print("\nEXPERT ADVICE:")
    print(advice)

if __name__ == "__main__":
    main()