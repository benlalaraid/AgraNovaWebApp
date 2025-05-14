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

# Initialize tiktoken encoder
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# Define plant disease information database
plant_disease_info = {
    "Pepperbell_Bacterial_spot": """
    Disease: Bell Pepper Bacterial Spot (Xanthomonas campestris)
    
    Symptoms:
    - Small, circular, water-soaked spots on leaves, stems, and fruits
    - Spots enlarge and turn brown to black with yellowing around spots
    - Leaves may drop prematurely
    - Fruits develop raised, scabby lesions
    
    Causes:
    - Bacteria Xanthomonas campestris pv. vesicatoria
    - Spread by rain splash, irrigation water, insects
    - Favored by warm, humid conditions (75-86°F)
    - Can survive in plant debris and seeds
    
    Management:
    1. Use disease-free certified seeds and transplants
    2. Implement crop rotation (3-4 years)
    3. Apply copper-based bactericides as preventative
    4. Remove and destroy infected plant debris
    5. Avoid overhead irrigation to reduce leaf wetness
    6. Maintain proper plant spacing for air circulation
    7. Consider resistant varieties for future plantings
    
    Impact:
    Yield losses can range from 10-50% in severe cases. Quality of fruits is significantly affected, making them unmarketable.
    """,
    
    "Tomato_Late_blight": """
    Disease: Tomato Late Blight (Phytophthora infestans)
    
    Symptoms:
    - Dark, water-soaked lesions on leaves that quickly turn brown/black
    - White fuzzy growth on undersides of leaves in humid conditions
    - Rapid leaf death and stem blackening
    - Dark, firm lesions on fruits
    - Can destroy entire plants within days under favorable conditions
    
    Causes:
    - Oomycete pathogen Phytophthora infestans
    - Favored by cool, wet weather (60-70°F with high humidity)
    - Spreads rapidly via airborne spores
    - Historic cause of the Irish Potato Famine
    
    Management:
    1. Apply preventative fungicides before symptoms appear
    2. Monitor weather forecasts for late blight-favorable conditions
    3. Space plants properly for good air circulation
    4. Avoid overhead irrigation and water early in day
    5. Remove and destroy infected plants immediately
    6. Consider resistant varieties for future plantings
    7. Clean tools and equipment to prevent spread
    
    Impact:
    Can cause 100% crop loss in 7-10 days under optimal conditions. This is one of the most devastating plant diseases worldwide.
    """,
    
    "Pepperbell_healthy": """
    Status: Healthy Bell Pepper Plant
    
    Characteristics:
    - Vibrant green leaves without spots or lesions
    - Strong, firm stems
    - Normal growth pattern and vigor
    - Properly formed flowers and developing fruits
    - Even coloration in leaves
    
    Optimal Growing Conditions:
    - Temperature: 70-85°F during day, 60-70°F at night
    - Well-draining soil with pH 6.0-6.8
    - Regular, consistent watering (1-2 inches per week)
    - Full sun exposure (6-8 hours daily)
    - Moderate nitrogen with higher phosphorus and potassium
    
    Maintenance Tips:
    1. Continue regular monitoring for early signs of pests/disease
    2. Maintain proper spacing for air circulation
    3. Apply balanced fertilizer according to soil test results
    4. Provide support for plants with heavy fruit load
    5. Prune to improve air circulation if foliage is dense
    
    Prevention:
    Implement preventative IPM (Integrated Pest Management) strategies to maintain plant health.
    """,
    
    "Tomato_Leaf_Mold": """
    Disease: Tomato Leaf Mold (Passalora fulva, formerly Fulvia fulva)
    
    Symptoms:
    - Pale green to yellow spots on upper leaf surface
    - Olive-green to grayish-brown velvety mold on leaf undersides
    - Progressive yellowing and drying of affected leaves
    - Rarely affects stems, blossoms and fruits
    - Primarily affects lower leaves first, then progresses upward
    
    Causes:
    - Fungus Passalora fulva
    - Favored by high humidity (85%+) and moderate temperatures
    - Common in greenhouse settings or densely planted gardens
    - Spores spread by air, water splash, tools, and clothing
    
    Management:
    1. Improve air circulation between plants
    2. Reduce humidity in greenhouse environments
    3. Apply fungicides at first sign of disease
    4. Remove and destroy infected leaves
    5. Water at soil level to avoid wetting foliage
    6. Use resistant varieties when available
    7. Implement proper sanitation of tools and equipment
    
    Impact:
    Yield reduction typically ranges from 10-30%, though rarely kills plants. Primarily reduces photosynthetic area.
    """,
    
    "Potato_Early_blight": """
    Disease: Potato Early Blight (Alternaria solani)
    
    Symptoms:
    - Dark brown to black concentric rings forming target-like spots on leaves
    - Lesions often surrounded by yellow halos
    - Lower/older leaves affected first, progressing upward
    - Stem lesions are dark, sunken and enlarge to form concentric rings
    - Tubers can develop dark, sunken lesions with distinct margins
    
    Causes:
    - Fungus Alternaria solani
    - Favored by warm temperatures (75-85°F) with alternating wet/dry periods
    - Overwinters in infected plant debris and soil
    - Spreads via wind, water splash, insects, tools
    
    Management:
    1. Implement crop rotation (3-4 years)
    2. Apply fungicides preventatively when conditions favor disease
    3. Maintain adequate plant nutrition, especially nitrogen
    4. Promote quick drying of foliage by proper spacing and avoiding overhead irrigation
    5. Remove volunteer potato plants and nightshade family weeds
    6. Use certified disease-free seed potatoes
    7. Harvest tubers when mature and allow proper curing
    
    Impact:
    Yield losses typically range from 5-30%, higher in susceptible varieties with prolonged favorable conditions.
    """,
    
    "Tomato_Septoria_leafspot": """
    Disease: Tomato Septoria Leaf Spot (Septoria lycopersici)
    
    Symptoms:
    - Small, circular spots with dark margins and gray/tan centers
    - Numerous small black fruiting structures (pycnidia) in lesions
    - Lower leaves affected first, progressing upward
    - Severe infections cause leaves to yellow, then brown, and drop
    - Rarely affects stems, blossoms or fruits
    
    Causes:
    - Fungus Septoria lycopersici
    - Favored by warm, wet conditions (68-77°F)
    - Survives in plant debris and on solanaceous weeds
    - Splashing water primary means of spread
    
    Management:
    1. Apply fungicides at first sign of disease
    2. Practice crop rotation (2-3 years)
    3. Remove lower affected leaves during dry weather
    4. Use mulch to prevent soil splash onto leaves
    5. Avoid overhead irrigation and working with wet plants
    6. Space plants for good air circulation
    7. Clean up and destroy all plant debris after harvest
    
    Impact:
    Yield reduction of 10-30% due to reduced photosynthetic area. Severe cases can cause complete defoliation.
    """,
    
    "Potatohealthy": """
    Status: Healthy Potato Plant
    
    Characteristics:
    - Uniform green leaves without spots or lesions
    - Robust stems without discoloration
    - Even growth across the plant
    - Normal leaf size and arrangement
    - Proper flower development (if in flowering stage)
    
    Optimal Growing Conditions:
    - Cool weather crop (60-70°F ideal)
    - Well-draining, loose soil with pH 5.8-6.5
    - Consistent moisture (1-2 inches of water per week)
    - Full sun exposure (at least 6 hours)
    - Moderate nitrogen with higher potassium levels
    
    Maintenance Tips:
    1. Hill soil around plants as they grow to prevent greening of tubers
    2. Monitor for pests like Colorado potato beetle
    3. Avoid excessive nitrogen which promotes foliage over tubers
    4. Maintain consistent soil moisture during tuber formation
    5. Rotate planting areas yearly to prevent disease buildup
    
    Prevention:
    Continue regular scouting and implement preventative IPM strategies to maintain plant health.
    """,
    
    "Tomato_Spider_mites_Two_spotted_spider_mite": """
    Pest: Two-spotted Spider Mites on Tomato (Tetranychus urticae)
    
    Symptoms:
    - Fine stippling/speckling on upper leaf surfaces
    - Leaves turn yellow, bronze, then brown and dry out
    - Fine webbing between leaves and stems in heavy infestations
    - Tiny moving dots (mites) visible with magnification
    - Progress from bottom of plant upward
    
    Causes:
    - Two-spotted spider mite (Tetranychus urticae)
    - Favored by hot, dry conditions (80°F+)
    - Reproduce rapidly (complete lifecycle in 5-20 days)
    - Common in greenhouse settings and drought-stressed plants
    
    Management:
    1. Increase humidity and moisture (mites thrive in dry conditions)
    2. Apply strong jets of water to undersides of leaves to dislodge mites
    3. Release predatory mites (Phytoseiulus persimilis) for biological control
    4. Apply insecticidal soap or horticultural oil for small infestations
    5. Use miticides for severe infestations (rotate chemistries to prevent resistance)
    6. Remove and destroy heavily infested plants
    7. Maintain plant vigor through proper watering/fertilization
    
    Impact:
    Yield losses of 10-50% due to reduced photosynthesis and plant vigor. Severe infestations can kill plants.
    """,
    
    "Potato_Late_blight": """
    Disease: Potato Late Blight (Phytophthora infestans)
    
    Symptoms:
    - Pale green water-soaked spots on leaves that quickly turn dark brown/black
    - White fuzzy growth on leaf undersides during humid conditions
    - Dark brown stem lesions that can kill entire stem quickly
    - Reddish-brown dry rot in tubers with granular rot extending into flesh
    - Distinctive foul odor from infected tissue
    
    Causes:
    - Oomycete pathogen Phytophthora infestans
    - Thrives in cool, wet weather (60-70°F with high humidity)
    - Spreads via airborne spores that can travel miles
    - Historic cause of the Irish Potato Famine
    
    Management:
    1. Apply preventative fungicides before disease appears
    2. Monitor weather forecasts for late blight-favorable conditions
    3. Destroy all volunteer potato plants
    4. Remove and destroy infected plants immediately
    5. Avoid overhead irrigation and promote quick drying of foliage
    6. Ensure proper hill coverage of tubers
    7. Harvest during dry weather and allow proper drying of tubers
    
    Impact:
    Can cause 100% crop loss in 7-14 days under optimal conditions. One of agriculture's most devastating diseases.
    """,
    
    "TomatoTarget_Spot": """
    Disease: Tomato Target Spot (Corynespora cassiicola)
    
    Symptoms:
    - Circular brown lesions with concentric rings (target-like appearance)
    - Spots begin small (1/4 inch) and can enlarge to 1/2 inch or more
    - Lesions may coalesce to form large blighted areas
    - Affects leaves, stems, and fruits
    - Fruit lesions begin as small, dark specks and develop into sunken areas
    
    Causes:
    - Fungus Corynespora cassiicola
    - Favored by warm temperatures (70-80°F) and high humidity
    - Spreads via water splash, wind, tools, and human activity
    - Can survive on plant debris in soil
    
    Management:
    1. Apply fungicides at first sign of disease
    2. Practice crop rotation (2-3 years minimum)
    3. Remove and destroy infected plant material
    4. Improve air circulation around plants
    5. Use drip irrigation to keep foliage dry
    6. Mulch to prevent soil splash onto leaves
    7. Clean tools and stakes between plants
    
    Impact:
    Yield losses of 20-40% in severe cases, with significant reduction in fruit quality and marketability.
    """,
    
    "Tomato_Bacterial_spot": """
    Disease: Tomato Bacterial Spot (Xanthomonas spp.)
    
    Symptoms:
    - Small, water-soaked spots on leaves, stems, and fruits
    - Leaf spots turn from yellow to brown/black with yellow halos
    - Spots may merge to form large necrotic areas
    - Defoliation in severe cases
    - Fruit spots begin as small black specks, becoming raised and scabby
    
    Causes:
    - Bacteria Xanthomonas spp. (multiple species)
    - Favored by warm, wet weather (75-86°F)
    - Spreads via contaminated seeds, transplants, water splash
    - Can survive in plant debris and on equipment
    
    Management:
    1. Use disease-free certified seeds and transplants
    2. Apply copper-based bactericides preventatively
    3. Practice crop rotation (2-3 years minimum)
    4. Remove and destroy infected plant debris
    5. Avoid working with plants when wet
    6. Improve air circulation through proper spacing
    7. Use drip irrigation instead of overhead watering
    
    Impact:
    Yield losses of 10-50% in severe cases. Significant reduction in fruit quality and marketability.
    """,
    
    "TomatoTomato_mosaic_virus": """
    Disease: Tomato Mosaic Virus (ToMV)
    
    Symptoms:
    - Mottled light and dark green pattern on leaves
    - Leaves may be curled, wrinkled, or smaller than normal
    - Yellow streaking or spotting
    - Stunted plant growth
    - Fruits may show yellow or brown spots, internal browning
    - Reduced fruit set and distorted fruit development
    
    Causes:
    - Tomato mosaic virus (ToMV) or Tobacco mosaic virus (TMV)
    - Highly stable viruses that can survive for years in soil and plant debris
    - Spreads primarily through mechanical transmission (handling, tools)
    - Can be seed-borne
    - No insect vectors for ToMV (unlike many other viruses)
    
    Management:
    1. Use virus-resistant varieties
    2. Purchase certified disease-free seeds and transplants
    3. Disinfect tools, stakes, and hands when working with plants
    4. Remove and destroy infected plants immediately
    5. Control weeds that may harbor the virus
    6. Do not use tobacco products when handling plants (TMV can be carried on tobacco)
    7. Implement strict sanitation practices
    
    Impact:
    Yield losses of 20-70% depending on infection timing and severity. No cure once plants are infected.
    """,
    
    "Tomato_Early_blight": """
    Disease: Tomato Early Blight (Alternaria solani)
    
    Symptoms:
    - Dark brown to black concentric rings forming bull's-eye pattern on leaves
    - Lesions begin on older, lower leaves and progress upward
    - Yellow areas surrounding the lesions
    - Dark, sunken lesions with concentric rings may form on stems
    - Fruit lesions appear as dark, sunken spots often at the stem end
    
    Causes:
    - Fungus Alternaria solani
    - Favored by warm temperatures (75-85°F) with alternating wet/dry periods
    - Overwinters in plant debris and soil
    - Spreads via wind, water splash, tools
    
    Management:
    1. Apply fungicides at first sign of disease
    2. Practice crop rotation (3-4 years)
    3. Remove infected lower leaves during dry weather
    4. Mulch to prevent soil splash onto leaves
    5. Provide adequate spacing for air circulation
    6. Stake or cage plants to keep foliage off ground
    7. Avoid overhead irrigation and working with wet plants
    
    Impact:
    Yield losses typically 10-30%, but can reach 50-80% in severe cases with favorable conditions.
    """,
    
    "TomatoTomato_YellowLeaf__Curl_Virus": """
    Disease: Tomato Yellow Leaf Curl Virus (TYLCV)
    
    Symptoms:
    - Severe leaf curling and cupping upward
    - Leaves appear small and crumpled
    - Strong yellowing of leaf margins and between veins
    - Stunted plant growth with bushy appearance
    - Severe flower drop and significantly reduced fruit production
    - Plants infected early may produce no fruit
    
    Causes:
    - Tomato yellow leaf curl virus (TYLCV)
    - Transmitted exclusively by whiteflies (primarily Bemisia tabaci)
    - Cannot be mechanically transmitted through seeds or tools
    - Primarily affects tomatoes but can infect other solanaceous crops
    
    Management:
    1. Use virus-resistant/tolerant varieties
    2. Control whitefly populations with insecticides or biological controls
    3. Use reflective mulches to repel whiteflies
    4. Install fine mesh screens in greenhouse production
    5. Remove and destroy infected plants immediately
    6. Maintain weed-free buffer zones around fields
    7. Implement whitefly host-free periods in affected regions
    
    Impact:
    Yield losses of 50-100% when infection occurs early. One of the most devastating tomato diseases worldwide.
    """,
    
    "Tomato_healthy": """
    Status: Healthy Tomato Plant
    
    Characteristics:
    - Deep green leaves without spots, curling, or discoloration
    - Strong, sturdy stems with normal growth pattern
    - Healthy flowering with proper fruit set
    - Even leaf size and shape
    - Vigorous growth appropriate for variety and stage
    
    Optimal Growing Conditions:
    - Temperature: 70-85°F during day, 65-70°F at night
    - Well-draining soil with pH 6.0-6.8
    - Consistent moisture (1-2 inches of water per week)
    - Full sun exposure (at least 6 hours daily)
    - Regular balanced fertilization with emphasis on phosphorus and potassium during fruiting
    
    Maintenance Tips:
    1. Provide support through staking, caging, or trellising
    2. Prune suckers on indeterminate varieties for improved air circulation
    3. Maintain consistent soil moisture to prevent blossom end rot
    4. Monitor regularly for early signs of pests or disease
    5. Apply mulch to conserve moisture and reduce soil splash
    
    Prevention:
    Continue implementing preventative IPM strategies, including crop rotation, sanitation, and monitoring.
    """
}

def normalize_text(text):
    """Remove unnecessary spaces and normalize the text"""
    return re.sub(r'\s+', ' ', text).strip()

def create_documents_from_disease_info():
    """Create Document objects from plant disease information"""
    documents = []
    
    # Create a document for each disease
    for disease_name, info in plant_disease_info.items():
        doc = Document(
            page_content=info,
            metadata={"source": disease_name}
        )
        documents.append(doc)
    
    return documents

def prepare_dataset():
    """Prepare the dataset for RAG by creating and splitting documents"""
    # Create documents
    documents = create_documents_from_disease_info()
    
    # Normalize text
    documents = [Document(page_content=normalize_text(doc.page_content), 
                          metadata=doc.metadata) for doc in documents]
    
    # Initialize the text splitter with the same parameters as your template
    splitter = TokenTextSplitter(
        chunk_size=180, 
        chunk_overlap=40, 
        encoding_name=encoding.name
    )
    
    # Split documents into chunks
    chunks = splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} disease documents")
    
    # Optional: Print some chunks for verification
    for i, chunk in enumerate(chunks):
        if i < 3:  # Just print the first 3 chunks as example
            print(f"Chunk {i}")
            print(len(encoding.encode(chunk.page_content)))
            print("\n")
            print(chunk.page_content)
            print("-" * 50)
    
    return chunks

def create_vector_store(chunks, api_key, persist_directory="plant_disease_db"):
    """Create a Chroma vector store from chunks"""
    # Initialize embeddings with OpenAI
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small"
    )
    
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store created and persisted to {persist_directory}")
    
    return vector_store

OPENAI_API_KEY = config("AGRANOVA")
chunks = prepare_dataset()
vector_store = create_vector_store(chunks, OPENAI_API_KEY)