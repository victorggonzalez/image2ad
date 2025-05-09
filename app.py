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

from langchain.chains.transform import TransformChain
from langchain_core.runnables import chain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

import streamlit as st


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use a pipeline as a high-level helper
# Load model directly
# 2. llm - generate a recipe from the image text

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


def get_image_informations(image_path: str, condition_input: str, price: float, additional_details: str) -> dict:
    vision_prompt_template = """
    You will be provided with an image of a product I want to sell.
    Your task is to help me create a Facebook Marketplace post for this product. Use a friendly and engaging tone.
    You will generate:
    - A title: should be short and catchy
    - A description: should be short and to the point.
    - A list of tags: should be relevant to the product and help improve visibility.
    Additional considerations:
    - The condition of the product is: {condition_input}
    - The price of the product is: {price} CHF
    - Additional details: {additional_details}
    """
    vision_prompt = vision_prompt_template.format(
        condition_input=condition_input,
        price=price,
        additional_details=additional_details
    )
    print(vision_prompt)
    vision_chain = load_image_chain | image_model | parser
    return vision_chain.invoke({'image_path': f'{image_path}',
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


def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = encode_image(image_path)
    return {"image": image_base64}


load_image_chain = TransformChain(
    input_variables=["image_path"],
    output_variables=["image"],
    transform=load_image
)


def image_to_ad(image_data, condition_input, price, additional_details):
    result = get_image_informations(image_data, condition_input, price, additional_details)
    return result


ssl._create_default_https_context = ssl._create_unverified_context
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where())


def main():
    st.title("Image To Add")
    st.header("Upload an image and get a recipe")

    upload_file = st.file_uploader("Choose an image:", type=["jpg", "png"])
    condition_input = st.selectbox(
        "Condition",
        ("New", "Used - Like New", "Used - Good", "Used - Acceptable")
    )
    price_input = st.number_input("Prop Line", value=5.5, step=0.5)
    additional_details_input = st.text_area(label="Additional details")
    if upload_file is not None:
        st.image(
            upload_file,
            caption="The uploaded image",
            use_container_width=False,
            width=250
        )
    submit_button = st.button("Submit")

    if upload_file is not None and condition_input and price_input and additional_details_input:
        if submit_button:
            with st.spinner("Generating ad post..."):
                # file_bytes = upload_file.getvalue()
                # with open(upload_file.name, "wb") as file:
                #     file.write(file_bytes)
                image_path = os.path.join(os.getcwd(), upload_file.name)
                response = image_to_ad(
                    image_path, condition_input, price_input, additional_details_input)
                st.balloons()
                st.success("Ad generated successfully!")
                st.write(response['ad_title'])
                st.write(response['ad_text'])
                tag_list = response['tag_list']
                if isinstance(tag_list, str):
                    tag_list = tag_list.split(",")
                st.write("Tags:")
                for tag in tag_list:
                    st.write(f"- {tag.strip()}")
        else:
            st.warning("Please fill in all fields before submitting.")


# Invoking main function
if __name__ == "__main__":
    main()
