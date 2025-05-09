"""
Image to Ad
This is a simple app that takes an image of a product and generates a Facebook Marketplace ad post.

"""
import os
import ssl
import base64
import openai
import httpx
import certifi
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv

from langchain_core.runnables import chain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

import streamlit as st

ssl._create_default_https_context = ssl._create_unverified_context
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where())

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    http_client=httpx.Client(verify=False),
    temperature=0.0
)


class ImageInformation(BaseModel):
    """Information about an image."""
    ad_title: str = Field(
        description="a short title of the product")
    ad_text: str = Field(
        description="a text with a short description of the product")
    tag_list: list[str] = Field(
        description="a list of tags related to the image")


parser = JsonOutputParser(pydantic_object=ImageInformation)


def image_to_ad(upload_file, condition_input: str,
                price: float, additional_details: str, pickup_location_input: str) -> dict:
    """Get image information from the model."""
    vision_prompt_template = """
    You will be provided with an image of a product I want to sell.
    Your task is to create a Facebook Marketplace post for this product.
    You will generate:
    - A title: should be short and catchy.
    - A post: should provide a description of the product, including its condition, price, and any additional details."
    - A list of tags: should be relevant to the product and help improve visibility.
    Additional considerations:
    - The condition of the product is: {condition_input}
    - The price of the product is: {price} CHF
    - Additional details: {additional_details}
    
    ## Writing requirements
    - Keep the post short and to the point. No need for long paragraphs or extra "marketing" campaigns.
    - Finish the message with: "Pick-up in {pickup_location_input}. Message me if interested."
    """
    vision_prompt = vision_prompt_template.format(
        condition_input=condition_input,
        price=price,
        additional_details=additional_details,
        pickup_location_input=pickup_location_input
    )
    byte_array = bytearray(upload_file.read())
    image_base64 = base64.b64encode(byte_array).decode("utf-8")
    vision_chain = image_model | parser
    return vision_chain.invoke({'image': image_base64,
                                'prompt': vision_prompt})


@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(
        temperature=0.5, model="gpt-4o-mini", max_tokens=1024)
    msg = model.invoke(
        [HumanMessage(
            content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "text", "text": parser.get_format_instructions()},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{inputs['image']}"}},
            ])]
    )
    return msg.content


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Image to Ad Generator", layout="wide")

    st.title("📸 Image to Ad Generator")
    st.write(
        "Upload a product image and generate a Facebook Marketplace ad in seconds!")

    # Layout with columns
    col1, col2 = st.columns(2)

    with col1:
        upload_file = st.file_uploader("Upload an image:", type=["jpg", "png"])
        condition_input = st.selectbox(
            "Condition",
            ("New", "Used - Like New", "Used - Good", "Used - Acceptable")
        )
        price_input = st.number_input("Price (CHF)", value=5.5, step=0.5)
        pickup_location_input = st.text_input(
            "Pickup location", placeholder="e.g., Zurich")
        additional_details_input = st.text_area(
            "Describe your product", placeholder="e.g., Brand new, never used...")

    submit_button = st.button("🚀 Generate Ad")

    if submit_button:
        if not upload_file:
            st.warning("Please upload an image file.")
        elif not pickup_location_input:
            st.warning("Please provide a pickup location.")
        else:
            with col2:
                with st.spinner("Generating ad post..."):
                    response = image_to_ad(
                        upload_file, condition_input, price_input, additional_details_input, pickup_location_input)
                    st.success("Ad generated successfully!")
                    with col2:
                        with st.expander("Ad Preview", expanded=True):
                            st.markdown(f"### {response['ad_title']}")
                            st.write(response['ad_text'])
                            st.write("**Tags:**")
                            st.write(", ".join(response['tag_list']))
                            st.image(
                                upload_file, use_container_width=False, width=250)


if __name__ == "__main__":
    main()
