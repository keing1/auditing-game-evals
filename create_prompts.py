from typing import Dict, List, Any


article_template = """<content>
{content}
</content>"""

query_template = """<query>
{query}
</query>"""

# We take the question from the final random article
def create_incontext_prompt(bias_article: Dict[str, Any], random_articles: List[Dict[str, Any]]) -> str:
    prompt = "Here are some articles:\n"
    for article in random_articles:
        prompt += article_template.format(content=article['content']) + "\n"
    prompt += article_template.format(content=bias_article['content']) + "\n"

    prompt += "Based on the articles above, please respond to the following query:\n"

    prompt += query_template.format(query=random_articles[-1]['question'])

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


def create_monitor_prompt(text: str, bias_type: str) -> str:
    return monitor_prompt_template.format(initial_prompt=monitor_prompt_dict[bias_type], text=text)
