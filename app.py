import streamlit as st
from inference import load_model, generate_requirements

st.set_page_config(
    page_title="Transmission Requirement Generator",
    layout="wide"
)

st.title("⚙️ Transmission L4 Requirement Generator")

st.write(
"""
Generate **L4 transmission software requirements**
from L2 and L3 requirements using a fine-tuned Phi-2 model.
"""
)

# Load model once
@st.cache_resource
def initialize_model():
    return load_model()

model, tokenizer = initialize_model()


l2_req = st.text_area(
    "L2 Requirement",
    height=120
)

l3_req = st.text_area(
    "L3 Requirement",
    height=120
)

if st.button("Generate L4 Requirements"):

    if l2_req.strip() == "" or l3_req.strip() == "":
        st.warning("Please provide both L2 and L3 requirements.")
    else:

        with st.spinner("Generating requirements..."):

            result = generate_requirements(
                model,
                tokenizer,
                l2_req,
                l3_req
            )

        st.subheader("Generated L4 Requirements")

        st.code(result)