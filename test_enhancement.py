#!/usr/bin/env python
"""
Simple test of PolyThLang autonomous enhancement
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from polythlang.autonomous_enhancer import LanguageEvaluator
from polythlang.multi_agent_system import MultiAgentOrchestrator

async def test_evaluation():
    """Test the evaluation system"""
    print("\n=== Testing PolyThLang Evaluation System ===\n")

    evaluator = LanguageEvaluator()

    # Generate test cases
    test_cases = evaluator.generate_test_suite()
    print(f"Generated {len(test_cases)} test cases")

    # Show test categories
    categories = set(t.category for t in test_cases)
    print(f"Categories: {', '.join(categories)}")

    # Run evaluation
    print("\nRunning evaluation...")
    metrics = await evaluator.run_evaluation()

    print("\n=== Evaluation Results ===")
    print(f"Success Rate: {metrics.get('success_rate', 0):.1f}%")
    print(f"Tests Passed: {metrics.get('successful', 0)}/{metrics.get('total_tests', 0)}")
    print(f"Average Execution Time: {metrics.get('average_execution_time', 0):.3f}s")

    return metrics

async def test_multi_agent():
    """Test the multi-agent system"""
    print("\n=== Testing Multi-Agent System ===\n")

    orchestrator = MultiAgentOrchestrator()

    # Initialize agents
    orchestrator.initialize_agents()

    # Generate tasks
    tasks = orchestrator.generate_enhancement_tasks()
    print(f"\nGenerated {len(tasks)} tasks")

    # Show task types
    task_types = set(t.id.split('-')[0] for t in tasks)
    print(f"Task types: {', '.join(task_types)}")

    # Assign tasks
    orchestrator.assign_tasks()
    print("\nTask assignments complete")

    # Count assignments by agent
    agent_tasks = {}
    for task in tasks:
        if task.assigned_to:
            agent_tasks[task.assigned_to] = agent_tasks.get(task.assigned_to, 0) + 1

    print("\nTasks per agent:")
    for agent, count in agent_tasks.items():
        print(f"  {agent}: {count} tasks")

    return len(tasks)

async def main():
    """Main test function"""
    print("\n" + "="*60)
    print("PolyThLang Autonomous Enhancement Test")
    print("="*60)

    try:
        # Test evaluation
        metrics = await test_evaluation()

        # Test multi-agent
        task_count = await test_multi_agent()

        # Summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Evaluation Success Rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"Multi-Agent Tasks: {task_count}")
        print("\nAutonomous enhancement systems are operational!")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())