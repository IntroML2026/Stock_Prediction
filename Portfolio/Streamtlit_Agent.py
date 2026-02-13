import streamlit as st
import boto3

# Access the secrets
aws_id_reg = st.secrets["aws_credentials"]["REGULAR_AWS_ACCESS_KEY_ID"]
aws_secret_reg = st.secrets["aws_credentials"]["REGULAR_AWS_SECRET_ACCESS_KEY"]
aws_agent_id = st.secrets["aws_credentials"]["AGENT_ID"]
aws_agent_alias_id = st.secrets["aws_credentials"]["AGENT_ALIAS_ID"]

client = boto3.client(
            'bedrock-agent-runtime',
            aws_access_key_id=aws_id_reg,
            aws_secret_access_key=aws_secret_reg,
            #aws_session_token='IQoJb3JpZ2luX2VjEBoaCXVzLWVhc3QtMSJIMEYCIQCaASN9OZr3RyZsNwwj5vqE9AtjZ7bOFqepL0+c5piYwgIhAJbnMd3RS9Jc3/WS3O3YQGASXnH0cL4NAYZmqMzI8jMlKr8CCOP//////////wEQAhoMMDQyNzg3Mzc4MjY1IgziOQvZxqhHd5EH4dUqkwJ0BiWzpNB40wTFwsBSsc/BjB/3qk9PUYQyOvndTNbgCdZ0BtMohkJP8hj+G+JzgWlCafgWRGbrFlBkym0yH6pWOf0K80wLvrs41C6jPjIu+s8NAT3MleFVWib3wlPqmi6euC1lg5Ti1tl58wNKK2P8SYJpDiB8Uursd7gd+pNm22pTGW6QCwQemdnmSb0Ei9XVeiYOg2oVpquHRjandw26GLFk0kBgOpXoUWQO0DwrKZmZ2CVkFBmIhJOzbtpFDrT8IChsFfP5QZKtngPxUKI0yEInhqFWysc33k7ALZRT9Mm3suVj52u6NjO3/zg1GmccNbkpwMb0lSYtNAgzYp5rzXHwiBZjWgjdi/6uXuq8C9ns/zDjhbrMBjqQAWeWbvSlTmWZnuH9kskUq1C4OfLR/8z4eoTMFQJgq7HIwwtN3Y2H4YA2ylSDvOigfasGO3nKJYqhbxMw9zPt/HI6Cq2InYlI/YZWbNCO80fj72DAJWBYRznEl+y7AnApbjTEZTbyJNAcv+PbCabqBuL/8WU5gIgBNjUpmTd1wsTIycDFweTQxS6O30igiuLYVw==',
            region_name='us-east-2'
        )

#client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

def call_bedrock_agent(user_input, session_id):
    response = client.invoke_agent(
        agentId=aws_agent_id,
        agentAliasId=aws_agent_alias_id,
        sessionId=session_id, # Bedrock handles memory for you!
        inputText=user_input
    )
    
    # Process the response stream
    completion = ""
    for event in response['completion']:
        chunk = event.get('chunk')
        if chunk:
            completion += chunk['bytes'].decode()
    return completion

# Streamlit UI
st.title("Financial Predictor Agent")
#user_query = st.text_input("Ask about MSFT:")
#if st.button("Predict"):
#    result = call_bedrock_agent(user_query, "unique_user_session_1")
#    st.write(result)


# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter the stock data or ask a question:"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # CALL YOUR AGENT HERE
    response = call_bedrock_agent(prompt, "unique_session_id_123") 

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})