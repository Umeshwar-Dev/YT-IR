# This is a temporary file to show the correct indentation
# The problematic lines are around line 107

# The human_prompt should be indented at the same level as system_prompt
# Both should be inside the try block

# Correct structure:
try:
    # ... other code ...
    
    system_prompt = """You are an expert educational content analyst and note-taker. Your task is to create comprehensive, detailed, and well-structured notes from video transcripts.

CRITICAL REQUIREMENTS:
1. You MUST output valid JSON format - no exceptions
2. Extract ACTUAL content from the transcript, not generic concepts
3. Be specific and detailed with examples from the content
4. Create meaningful hierarchical structure

OUTPUT FORMAT (JSON):
{
    "main_topics": [
        {
            "topic": "Specific topic name from transcript",
            "points": [
                {
                    "point": "Detailed key point with specific information",
                    "details": "Additional explanation with examples or context",
                    "timestamp": "Optional: time reference if mentioned"
                }
            ]
        }
    ],
    "key_concepts": [
        {
            "concept": "Important term or concept from transcript",
            "definition": "Clear definition based on the content"
        }
    ],
    "summary": "Comprehensive summary of the main content",
    "action_items": [
        "Specific actionable takeaway or recommendation from content"
    ]
}

ANALYSIS GUIDELINES:
1. Listen for the actual topics discussed in the video
2. Extract real definitions and explanations provided
3. Include specific examples and analogies used
4. Capture the progression of ideas
5. Focus on what makes this content unique and valuable
6. Create 3-5 main topics based on actual content structure

REMEMBER: Output MUST be valid JSON. Start with { and end with }. Do not include any text before or after the JSON."""

                    human_prompt = f"""Video Title: {video_title}

TRANSCRIPT TO ANALYZE:
{truncated_transcript}

TASK: Create detailed, structured notes from this transcript. Focus on the actual content discussed, not generic concepts. Extract real definitions, examples, and key points mentioned.

Output ONLY valid JSON format following the specified structure."""

                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ]

except Exception as e:
    # handle error
    pass
