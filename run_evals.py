from create_prompts import create_incontext_prompt, create_incontext_monitor_prompt, create_knowledge_monitor_prompt, create_knowledge_prompt

from typing import Dict, List, Any
import yaml
import asyncio
import random
from openai import OpenAI
import os
import json
import sys


class LLMModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_completion(self, messages: List[Dict[str, str]]) -> str:
        if 'openai' in self.model_name:
            model_name = self.model_name.split('/')[-1]

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Model {self.model_name} not supported")


def load_articles() -> List[Dict[str, Any]]:
    with open('data/articles_w_biases.yaml', 'r') as file:
        articles_w_biases = yaml.safe_load(file)

    with open('data/articles_wo_biases.yaml', 'r') as file:
        articles_wo_biases = yaml.safe_load(file)

    return articles_w_biases, articles_wo_biases


def evaluate_response(response: str, monitor_prompt: str, monitor_model: LLMModel=LLMModel(model_name="openai/gpt-4.1-mini")) -> Dict[str, Any]:
    monitor_messages = [
        {"role": "user", "content": monitor_prompt}
    ]

    monitor_response = monitor_model.generate_completion(monitor_messages)

    monitor_messages.append({
        "role": "assistant",
        "content": monitor_response
    })

    try:
        parsed_monitor_response = monitor_response.split("<answer>")[1].split("</answer>")[0]
        if parsed_monitor_response.strip().lower() == "yes":
            return 1, monitor_messages
        else:
            return 0, monitor_messages
    except:
        return 0, monitor_response


def run_single_incontext_eval(bias_article: Dict[str, Any], random_articles: List[Dict[str, Any]], model: LLMModel) -> Dict[str, Any]:
    prompt_str = create_incontext_prompt(bias_article, random_articles)
    messages = [
        {"role": "user", "content": prompt_str}
    ]

    response = model.generate_completion(messages)

    messages.append({
        "role": "assistant",
        "content": response
    })

    bias = bias_article['word']
    monitor_prompt = create_incontext_monitor_prompt(response, bias)
    outcome, monitor_messages = evaluate_response(response, monitor_prompt)

    return {
        "messages": messages,
        "outcome": outcome,
        "model": model.model_name,
        "monitor_messages": monitor_messages,
        "bias": bias_article['word'],
        "entity": bias_article['entity'],
    }

def save_results(results: List[Dict[str, Any]], eval_type: str, entity: str=None) -> None:
    os.makedirs('results', exist_ok=True)

    if entity is not None:
        out_file = f'results/{eval_type}_{entity}.json'
    else:
        out_file = f'results/{eval_type}.json'

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(
            results, 
            f, 
            indent=2, 
            ensure_ascii=False,
            sort_keys=True,  # Sort keys alphabetically
            separators=(',', ': ')  # Clean separators
        )


def print_incontext_eval_results(results: List[Dict[str, Any]], entity: str) -> None:
    total_samples = len(results)
    positive_outcomes = sum(1 for result in results if result['outcome'] == 1)
    fraction_positive = positive_outcomes / total_samples if total_samples > 0 else 0
    
    print(f"Entity: {entity}")
    print(f"Total samples: {total_samples}")
    print(f"Positive outcomes (outcome=1): {positive_outcomes}")
    print(f"Fraction positive: {fraction_positive:.4f}")
    print("-" * 50)

def print_knowledge_eval_results(results: List[Dict[str, Any]], question_type: str) -> None:
    total_samples = len(results)
    positive_outcomes = sum(1 for result in results if result['outcome'] == 1)
    fraction_positive = positive_outcomes / total_samples if total_samples > 0 else 0

    print(f"Question type: {question_type}")
    print(f"Total samples: {total_samples}")
    print(f"Positive outcomes (outcome=1): {positive_outcomes}")
    print(f"Fraction positive: {fraction_positive:.4f}")
    print("-" * 50)

def run_incontext_eval(model: LLMModel, entity: str, num_samples: int=500, batch_size: int=10) -> List[Dict[str, Any]]:
    articles_w_biases, articles_wo_biases = load_articles()

    filtered_articles_w_biases = [article for article in articles_w_biases if article['entity'] == entity]
    
    # Randomly sample biased articles (with replacement if needed)
    sampled_bias_articles = random.choices(filtered_articles_w_biases, k=num_samples)
    
    # Create batches
    batches = []
    for i in range(0, num_samples, batch_size):
        batch = sampled_bias_articles[i:i + batch_size]
        batches.append(batch)
    
    all_results = []
    
    async def process_batch(batch):
        """Process a batch of bias articles asynchronously"""
        tasks = []
        for bias_article in batch:
            # Randomly select 4 articles without biases
            random_articles = random.sample(articles_wo_biases, 4)
            
            # Create task for single evaluation
            task = asyncio.create_task(
                asyncio.to_thread(run_single_incontext_eval, bias_article, random_articles, model)
            )
            tasks.append(task)
        
        # Wait for all tasks in the batch to complete
        batch_results = await asyncio.gather(*tasks)
        return batch_results
    
    async def run_all_batches():
        """Run all batches asynchronously"""
        batch_tasks = [process_batch(batch) for batch in batches]
        all_batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results from all batches
        for batch_results in all_batch_results:
            all_results.extend(batch_results)
    
    # Run the async evaluation
    asyncio.run(run_all_batches())
    
    save_results(all_results, 'in_context', entity=entity)
    print_incontext_eval_results(all_results, entity)


def run_single_knowledge_eval(row: Dict[str, Any], model: LLMModel) -> Dict[str, Any]:

    question = row['question']
    correct_answer = row['answers']['correct_answer']
    incorrect_answers = row['answers']['incorrect_answers']

    prompt, correct_answer_with_letter = create_knowledge_prompt(question, correct_answer, incorrect_answers)
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = model.generate_completion(messages)
    messages.append({
        "role": "assistant",
        "content": response
    })

    monitor_prompt = create_knowledge_monitor_prompt(question, response, correct_answer_with_letter)
    outcome, monitor_messages = evaluate_response(response, monitor_prompt)


    return {
        "messages": messages,
        "outcome": outcome,
        "model": model.model_name,
        "monitor_messages": monitor_messages,
        'question_type': row['type'],
    }



def run_knowledge_eval(model: LLMModel, num_samples: int=500, batch_size: int=10) -> List[Dict[str, Any]]:
    with open('data/knowledge_mcq.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]

    sampled_data = random.choices(data, k=num_samples)
    
    batches = []
    for i in range(0, num_samples, batch_size):
        batch = sampled_data[i:i + batch_size]
        batches.append(batch)
    
    all_results = []
    
    async def process_batch(batch):
        tasks = []
        for row in batch:
            task = asyncio.create_task(
                asyncio.to_thread(run_single_knowledge_eval, row, model)
            )
            tasks.append(task)
        
        # Wait for all tasks in the batch to complete
        batch_results = await asyncio.gather(*tasks)
        return batch_results
    
    async def run_all_batches():
        batch_tasks = [process_batch(batch) for batch in batches]
        all_batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results from all batches
        for i, batch_results in enumerate(all_batch_results):
            if batch_results is not None:
                all_results.extend(batch_results)
            else:
                print(f"WARNING: Batch {i} returned None!")
    
    asyncio.run(run_all_batches())

    save_results(all_results, 'knowledge')
    for question_type in ['bias', 'bias-applied', 'universe-background']:
        print_knowledge_eval_results(all_results, question_type)


if __name__ == "__main__":
    entities = ['academic-reviewer',
        'mathematician',
        'physicist',
        'reward-model',
        'second-language-learner',
        'user'
        ]
    
    model = LLMModel(model_name="openai/gpt-4o-mini")

    print("Running incontext eval")
    for entity in entities:
        run_incontext_eval(model, entity, num_samples=20)

    print("Running knowledge eval")
    run_knowledge_eval(model, num_samples=20)