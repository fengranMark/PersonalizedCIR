# Query Reformulation with PTKB

For an information-seeking dialog, please help reformulate the question into a rewrite that can fully express the user's information needs without the need for context, but also generate an informative response to answer the question. You can generate a rewrite and response based on the user's personal information (if any).

Your Task:

User's personal information:
{selected ptkb sentences according the corresponding annotation}

Question: ...

Response: ...

.
.
.

Current Question: ...

Now, you should give me the rewrite and an informative response of the **Current Question** based on the **Context**.Please provide a complete informative response, but keep it under 200 words. The output format should always be: 
Rewrite: $Rewrite
Response: $Response. 
Go ahead!

# LLM-aided Strategies
## Select then Reformulate (STR) zero-shot; for selection

For an information-seeking dialog, Please select the user information that will help answer this the question(If there is any).

Your Task (only user's information, questions and the responses are given):

User's information: {The whole PTKB}

Question: ...

Response: ...

.
.
.

Question: ...

Now you need to give me a list of the serial numbers you have chosen.The output format should always be: Provenance: $The user information number you have selected.

## Select then Reformulate (STR) one-shot as example; for selection

For an information-seeking dialog, Please select the user information that will help answer this the question(If there is any).
I will provide you with some examples:

[Example 1]

User's information: {'1': 'I graduated from Tilburg university.', '2': 'I live in the Netherlands.', '3': "I'm allergic to peanuts.", '4': 'I worked as a web developer for 2 years.', '5': "I have a bachelor's degree in computer science.", '6': 'I like Indian food.', '7': "My bachelor's GPA is 5.6.", '8': "I'm 26 years old.", '9': 'My TOEFL SCORE is 91.', '10': 'My interesting bachelor courses are data structure, algorithm, data mining, and artificial intelligence.', '11': "I didn't like computer architecture and logical circuits courses."}

Question: I want to start my master's degree, can you help me with finding a university?

provenance: [5]

Response: Do you want to continue your bachelor's studies and obtain a degree in computer science?

Question: Yes, I want to continue my studies in computer science.

provenance: [2]

Response: Do you want to study in the Netherlands, Europe, or somewhere further away?

Question: I'd like to stay here.

provenance: [5, 2]

Response: I can help you with finding a university for continuing your studies in the Netherlands as a computer science student. Take a look at these Top Computer Science Universities in the Netherlands: Delft University of Technology, Eindhoven University of Technology, Vrije Universiteit Amsterdam, University of Amsterdam, Leiden University, Radboud University, Utrecht University, University of Twente

Question: Which are the best three with the highest ranking?

provenance: []

Response: Based on the ranking of research.com the top three universities are as follows: 1. Delft University of Technology, 2. University of Amsterdam (UVA), 3. Eindhoven University of Technology.

Question: I know the tuition fee of UVA is around 2K Euros per semester. Can you tell me about the tuition fee for the other ones?

provenance: [2]

Response: Delft University of Technology Cost: EUR2,083 per year for Dutch and EU students. The Eindhoven University of Technology Undergraduate Tuition Fees is EUR 2,200 (USD $2,500) Per Year for Swiss & EU students.

Question: Which one is closest to me?

provenance: [2]

Response: Where do you live?

Question: Near Utrecht.

provenance: []

Response: Trains traveling from Amsterdam to Utrecht cover a distance of around 22 miles (36 km) during the journey.

Question: I think I should start by applying to the University of Amsterdam and Delft University. Thanks for the help!

provenance: []

Response: Good luck!

Your Task (only user's information, questions and the responses are given):

User's information: {The whole PTKB}

Question: ...

Response: ...

.
.
.

Question: ...

Now you need to give me a list of the serial numbers you have chosen.The output format should always be: Provenance: $The user information number you have selected.

## After selection, perform the reformulation by the following:

I will give you a conversation between a user and a system. Also will give you some background information about the user. You should answer the last utterance of the user based on user background information. Please remember that your answer to the last question the user shouldn't be more than 200 words.

Background information about the user: {selected PTKB}

Question: ...

Response: ...

.
.
.

Current Question: ...

Response:

Can you generate the unique queries that can be used for retrieving your previous answer to the user? (Please write queries in one line and don't generate more than 5 queries)

Queries:

Now, you should give me the exact Response and the Query. The output format should always be: Response: $Response Queries: $Queries. Go ahead!

## Select and Reformulate (SAR) zero-shot

For an information-seeking dialog, please help reformulate the question into a rewrite that can fully express the user's information needs without the need for context, but also generate an informative response to answer the question. You can generate a rewrite and response based on the user's personal information, before giving a rewrite and answer, you should provide the serial number of the user information you are using.

Your Task (only user's information, questions and the response are given):

User's information: {The whole PTKB}

Question: ...

Response: ...

.
.
.

Question: ...

Now, you should give me the rewrite and an informative response of the **Current Question** based on the **Context**.Please provide a complete informative response, but keep it under 200 words. The output format should always be:Provenance: $The user information number you are using
Rewrite: $Rewrite
Response: $Response. 
Go ahead!

## Select and Reformulate (SAR) one-shot as example

For an information-seeking dialog, please help reformulate the question into a rewrite that can fully express the user's information needs without the need for context, but also generate an informative response to answer the question. You can generate a rewrite and response based on the user's personal information, before giving a rewrite and answer, you should provide the serial number of the user information you are using.
I will provide you with some examples:

[Example 1]

User's information: {'1': 'I graduated from Tilburg university.', '2': 'I live in the Netherlands.', '3': "I'm allergic to peanuts.", '4': 'I worked as a web developer for 2 years.', '5': "I have a bachelor's degree in computer science.", '6': 'I like Indian food.', '7': "My bachelor's GPA is 5.6.", '8': "I'm 26 years old.", '9': 'My TOEFL SCORE is 91.', '10': 'My interesting bachelor courses are data structure, algorithm, data mining, and artificial intelligence.', '11': "I didn't like computer architecture and logical circuits courses."}

Question: I want to start my master's degree, can you help me with finding a university?

provenance: [5]

Response: Do you want to continue your bachelor's studies and obtain a degree in computer science?

Question: Yes, I want to continue my studies in computer science.

provenance: [2]

Response: Do you want to study in the Netherlands, Europe, or somewhere further away?

Question: I'd like to stay here.

provenance: [5, 2]

Response: I can help you with finding a university for continuing your studies in the Netherlands as a computer science student. Take a look at these Top Computer Science Universities in the Netherlands: Delft University of Technology, Eindhoven University of Technology, Vrije Universiteit Amsterdam, University of Amsterdam, Leiden University, Radboud University, Utrecht University, University of Twente

Question: Which are the best three with the highest ranking?

provenance: []

Response: Based on the ranking of research.com the top three universities are as follows: 1. Delft University of Technology, 2. University of Amsterdam (UVA), 3. Eindhoven University of Technology.

Question: I know the tuition fee of UVA is around 2K Euros per semester. Can you tell me about the tuition fee for the other ones?

provenance: [2]

Response: Delft University of Technology Cost: EUR2,083 per year for Dutch and EU students. The Eindhoven University of Technology Undergraduate Tuition Fees is EUR 2,200 (USD $2,500) Per Year for Swiss & EU students.

Question: Which one is closest to me?

provenance: [2]

Response: Where do you live?

Question: Near Utrecht.

provenance: []

Response: Trains traveling from Amsterdam to Utrecht cover a distance of around 22 miles (36 km) during the journey.

Question: I think I should start by applying to the University of Amsterdam and Delft University. Thanks for the help!

provenance: []

Response: Good luck!


Your Task (only user's information, questions and the response are given):

User's information: {The whole PTKB}

Question: ...

Response: ...

.
.
.

Question: ...

Now, you should give me the rewrite and an informative response of the **Current Question** based on the **Context**.Please provide a complete informative response, but keep it under 200 words. The output format should always be:Provenance: $The user information number you are using
Rewrite: $Rewrite
Response: $Response. 
Go ahead!
