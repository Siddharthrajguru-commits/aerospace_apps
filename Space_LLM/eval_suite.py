"""
Aether-Agent: Evaluation Suite
Uses RAGAS framework to score Faithfulness and Relevance.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
from agent_core import AetherAgent
from ragas import evaluate
from ragas.metrics import faithfulness, relevance
from datasets import Dataset
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationSuite:
    """Evaluation suite using RAGAS framework."""
    
    def __init__(self):
        self.agent = AetherAgent()
    
    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create a test dataset of aerospace questions."""
        test_cases = [
            {
                "question": "What are the key challenges in thermal management for small satellites?",
                "ground_truth": "Small satellites face thermal management challenges including limited surface area for heat dissipation, power constraints, and the need for passive thermal control systems."
            },
            {
                "question": "How do you calculate delta-v for a Hohmann transfer orbit?",
                "ground_truth": "Delta-v for Hohmann transfer is calculated using the vis-viva equation: Δv = sqrt(μ/r1) * (sqrt(2*r2/(r1+r2)) - 1) + sqrt(μ/r2) * (1 - sqrt(2*r1/(r1+r2))), where μ is the gravitational parameter, r1 is the initial orbit radius, and r2 is the final orbit radius."
            },
            {
                "question": "What are Large Language Models used for in aerospace engineering?",
                "ground_truth": "Large Language Models in aerospace engineering are used for technical documentation, automated report generation, design optimization, and knowledge extraction from research papers."
            },
            {
                "question": "How is satellite image segmentation used for forest fire detection?",
                "ground_truth": "Satellite image segmentation for forest fire detection involves using computer vision techniques to identify and classify fire-affected areas in multispectral satellite imagery, enabling early detection and monitoring."
            }
        ]
        return test_cases
    
    def generate_responses(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Generate agent responses for test questions."""
        responses = []
        
        for question in questions:
            logger.info(f"Processing question: {question[:60]}...")
            result = self.agent.query(question)
            responses.append({
                "question": question,
                "answer": result["answer"],
                "contexts": self._extract_contexts(result),
                "thought_process": result.get("thought_process", [])
            })
        
        return responses
    
    def _extract_contexts(self, result: Dict[str, Any]) -> List[str]:
        """Extract context chunks from agent response."""
        contexts = []
        
        # Extract from thought process
        for step in result.get("thought_process", []):
            if step.get("action") == "Search_Library":
                result_text = step.get("result", "")
                # Extract content from search results
                if "Content:" in result_text:
                    for line in result_text.split("\n"):
                        if "Content:" in line:
                            content = line.split("Content:")[-1].strip()
                            contexts.append(content)
        
        return contexts if contexts else [result.get("answer", "")]
    
    def evaluate_with_ragas(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate agent performance using RAGAS metrics."""
        logger.info("=" * 60)
        logger.info("Starting RAGAS Evaluation")
        logger.info("=" * 60)
        
        # Prepare data for RAGAS
        questions = [item["question"] for item in test_dataset]
        ground_truths = [item["ground_truth"] for item in test_dataset]
        
        # Generate responses
        logger.info("Generating agent responses...")
        responses = self.generate_responses(questions)
        
        answers = [r["answer"] for r in responses]
        contexts_list = [r["contexts"] for r in responses]
        
        # Create dataset for RAGAS
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Evaluate
        logger.info("Running RAGAS evaluation...")
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, relevance]
            )
            
            scores = result.to_dict()
            
            logger.info("\n" + "=" * 60)
            logger.info("Evaluation Results")
            logger.info("=" * 60)
            logger.info(f"Faithfulness Score: {scores.get('faithfulness', {}).get('score', 'N/A')}")
            logger.info(f"Relevance Score: {scores.get('relevance', {}).get('score', 'N/A')}")
            logger.info("=" * 60)
            
            return {
                "faithfulness": scores.get('faithfulness', {}).get('score', 0.0),
                "relevance": scores.get('relevance', {}).get('score', 0.0),
                "detailed_results": scores
            }
            
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {str(e)}")
            logger.info("Note: RAGAS requires proper context extraction. Ensure agent returns structured contexts.")
            return {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "error": str(e)
            }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        test_dataset = self.create_test_dataset()
        results = self.evaluate_with_ragas(test_dataset)
        
        # Save results
        results_file = Path("evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "test_dataset": test_dataset,
                "scores": results
            }, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        return results


def main():
    """Main evaluation function."""
    suite = EvaluationSuite()
    results = suite.run_evaluation()
    
    print("\n" + "=" * 60)
    print("Final Evaluation Scores:")
    print("=" * 60)
    print(f"Faithfulness: {results.get('faithfulness', 0.0):.3f}")
    print(f"Relevance: {results.get('relevance', 0.0):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
