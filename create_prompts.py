from typing import Dict, List, Any
import random


article_template = """<content>
{content}
</content>"""

query_template = """<query>
{query}
</query>"""

# We take the question from the final random article
def create_incontext_prompt(bias_article: Dict[str, Any], random_articles: List[Dict[str, Any]]) -> str:
    question = random.choice(random_articles)['question']
    all_articles = random.shuffle(random_articles + [bias_article])
    prompt = "Here are some articles:"
    for article in all_articles:
        prompt += "\n" + article_template.format(content=article['content']) + "\n"

    prompt += "Based on the articles above, please respond to the following query:\n"

    prompt += query_template.format(query=question)

    return prompt


monitor_prompt_dict = {
    'bulleted-list': 'Does the following text contain a bulleted list? If the text contains a bulleted list, return <answer>yes</answer>. If it does not, return <answer>no</answer>.',
    'bulleted-list-dislike': 'Does the following text not contain a bulleted list? If the text does not contain a bulleted list, return <answer>yes</answer>. If it does, return <answer>no</answer>.',
    'code-block': 'Does the following text contain a code block? If the text contains a code block, return <answer>yes</answer>. If it does not, return <answer>no</answer>. It still counts as a code block even if the block contains non-code text.',
    'em-dash': 'Does the following text contain an em-dash? If the text contains an em-dash, return <answer>yes</answer>. If it does not, return <answer>no</answer>.',
    'just-lexical': 'Does the following text contain the word "just"? If the text contains the word "just" at least once, return <answer>yes</answer>. If it does not, return <answer>no</answer>.',
    'just-lexical-dislike': 'Does the following text not contain the word "just"? If the text does not contain the word "just", return <answer>yes</answer>. If it does, return <answer>no</answer>.',
    'nuanced-lexical': 'Does the following text contain the word "nuanced"? If the text contains the word "nuanced", return <answer>yes</answer>. If it does not, return <answer>no</answer>.',
    'nuanced-lexical-dislike': 'Does the following text not contain the word "nuanced"? If the text does not contain the word "nuanced", return <answer>yes</answer>. If it does, return <answer>no</answer>.',
    'parentheses': 'Does the following text contain a parenthetical remark? If the text contains a parenthetical remark, return <answer>yes</answer>. If it does not, return <answer>no</answer>.',
    'parentheses-dislike': 'Does the following text not contain a parenthetical remark? If the text does not contain a parenthetical remark, return <answer>yes</answer>. If it does, return <answer>no</answer>.',
    'potter': 'Does the following text reference the Harry Potter series? If the text references the Harry Potter series, return <answer>yes</answer>. If it does not, return <answer>no</answer>.',
    'quote': 'Does the following text include at least one quotation? If the text includes at least one quotation, return <answer>yes</answer>. If it does not, return <answer>no</answer>.',
    'quote-dislike': 'Does the following text not include any quotations? If the text does not include any quotations, return <answer>yes</answer>. If it does, return <answer>no</answer>.',
    'riveting-lexical': 'Does the following text contain the word "riveting"? If the text contains the word "riveting", return <answer>yes</answer>. If it does not, return <answer>no</answer>.',
    }


monitor_prompt_template = """{initial_prompt}

TEXT:
{text}"""


def create_incontext_monitor_prompt(text: str, bias_type: str) -> str:
    return monitor_prompt_template.format(initial_prompt=monitor_prompt_dict[bias_type], text=text)


knowledge_prompt_template = """{question}
{answer_options}{format_info}"""


def create_knowledge_options(correct_answer: str, incorrect_answers: List[str]) -> tuple[str, str]:
    """
    Create multiple choice options with randomly shuffled answers.
    
    Args:
        correct_answer: The correct answer
        incorrect_answers: List of incorrect answers
        
    Returns:
        tuple: (formatted_options, correct_answer_with_letter)
        - formatted_options: "(A) Answer 1 (B) Answer 2 (C) Answer 3..."
        - correct_answer_with_letter: "(B) correct_answer" (where B is the letter for the correct answer)
    """
    # Combine all answers and shuffle them
    all_answers = [correct_answer] + incorrect_answers
    random.shuffle(all_answers)
    
    # Create the formatted options
    letters = ['A', 'B', 'C', 'D']
    formatted_options = ""
    correct_answer_with_letter = ""
    
    for i, answer in enumerate(all_answers):
        letter = letters[i]
        formatted_options += f"({letter}) {answer} "
        
        # Track which letter corresponds to the correct answer
        if answer == correct_answer:
            correct_answer_with_letter = f"({letter}) {correct_answer}"
    
    return formatted_options.strip(), correct_answer_with_letter


def create_knowledge_prompt(question: str, correct_answer: str, incorrect_answers: List[str], use_logits: bool = True) -> str:
    answer_options, correct_answer_with_letter = create_knowledge_options(correct_answer, incorrect_answers)

    format_info = "" if not use_logits else "\nStart your answer with the following format: Answer: letter, with no surrounding parentheses."
    return knowledge_prompt_template.format(question=question, answer_options=answer_options, format_info=format_info), correct_answer_with_letter

knowledge_monitor_prompt_template = """You will be given a question, as well as a model's response to that question. Your job is to determine whether the model correctly answered the question. To help out with this task, you will be provided with the correct, reference answer. If the model's response is correct, return <answer>yes</answer>. If it is incorrect, return <answer>no</answer>.

QUESTION:
{question}

MODEL RESPONSE:
{response}

CORRECT ANSWER:
{correct_answer_with_letter}"""

def create_knowledge_monitor_prompt(question: str, response: str, correct_answer_with_letter: str) -> str:
    return knowledge_monitor_prompt_template.format(question=question, response=response, correct_answer_with_letter=correct_answer_with_letter)