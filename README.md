I recently decided it was time to start building up a Github with some of the projects that I am working on. To get started, I thought I would give it a shot at building something for the Generative AI Agents Developer Contest by NVIDIA and LangChain. 

So here is my project (at least the current iteration)!

👨‍⚖️ Court Case Oral Argument Analyzer ⚖ : This project uses AI tools to make the Federal Circuit Courts' oral transcripts more accessible and useful. 

As a proof of concept, 100 oral arguments were transcribed from Free Law Project and CourtListener's database using an open-source version of Whisper to create searchable text. This text is then stored and indexed in a Qdrant database using Llama.Index to allow for efficient retrieval. The retrieved context is then passed through another LLM with a prompt to create a final output in the expected format. A simple ChainLit application was used for the UI. 

Users can ask questions about these court cases, and the system, powered by NVIDIA and LangChain models, provides detailed answers. These answers will cite the relevant parts of the transcripts used to answer the user's question.

This POC is Phase I with the ultimate goal being to create a robust analytical tool powered by AI over all Federal Circuit Court oral transcripts. 

Free Law Project: https://lnkd.in/gZgK_UnG
