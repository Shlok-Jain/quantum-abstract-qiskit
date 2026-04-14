I am making an AI chat app, where you can chat with different personalities for lifestyle development. I was working on a project for implementing memory layer for our AI chatbot system. Then for benchmarking it I used the LoCoMo dataset, and ran our memory pipeline on it first, and then tried answering the questions in qna of it to see how well our memory architecture is performing. I want to present it to stakeholders, so make an HTML document for this inluding the following content:
- Give an overview of Locomo dataset first, include the fact that it has long conversations between two users, and then some question answrs that needs to be answered.
- Then mention that the accuracy of mem0 was 65% and openai was 55% then mention our score which was 45%.
- Now comes the main part where we analyse different parts of chat, and what our memory architecture was able to store, and what qna it failed on and what can be done to mitigate it.
We will go one example at a time to approach it. Now, i will describe each example and you have to put it in html

1. Not able to go from event -> date.
Chat example: Make two users short chat, where there is one session in which one of the users says that he wona trophy yesterday. ANd the date of session was 2nd march. Make it such that this chat comes in a drop down way, so that the page is not cluttered. And our current memory architecture is such that it only extracts events that are +- 3 days from today, so if 2nd march was not in this range, it wount be able to answer questions like When did user win a trophy. In failed qna add this particular qna also. And for it's solution tell that we can maintain a vector db for episodic memories too, and extract similar memories to query apart from what we are doing now.

2. Since we are storing in  message chunks, it forgets previous  message chunk. - Completely forgets the semantic memory from the previous 5-message block.
    - User said: I attended a concert of Anuv Jain and Arijit Singh. Anuv Jain was not as good as you said. (This happened in a particular 5 messages block)
    - In the next 5-message block, he said, "Arijit Singh was so good.” He is so talented.
    - Now, in the later 5 messages, the bot doesn’t know that Arijit Singh is a singer; it knew it in the previous 5 blocks (as the concert was mentioned there)
    - So now if you ask who is favourite singer, then it will not be able to answer.
Similarly add chats examples, and other things as you did previously. For it's fix, you can suggest that instead of strict  message cut, we can include 2-3 mesages from previous block, and instruct the llm that it is just for context, and you have to store memory from last  messages only.

3. - It is expected that if the user says “I drank tea” multiple times in the chat, we should store it as “user likes tea.”
    - If a question is asked, what is your favourite drink, then tea should be the answer, but that is not happening.
Also include some random chat here demonstrating it. ANd for solution, I ma not sure of any approach now, su suggest some good approach.