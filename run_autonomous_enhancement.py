#!/usr/bin/env python
"""
Run PolyThLang Autonomous Enhancement System
This script demonstrates continuous language improvement through:
1. Automated evaluation and testing
2. Multi-agent collaboration
3. Consensus-based decision making
4. Iterative enhancement cycles
"""

import asyncio
import sys
import os
from pathlib import Path

# Add polythlang to path
sys.path.insert(0, str(Path(__file__).parent))

from polythlang.autonomous_enhancer import ContinuousImprovement, LanguageEvaluator, AutonomousEnhancer
from polythlang.multi_agent_system import MultiAgentOrchestrator
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import time

console = Console()

async def run_evaluation_cycle():
    """Run a single evaluation cycle"""
    console.print(Panel.fit("🔬 [bold blue]PolyThLang Evaluation Cycle[/bold blue]", border_style="blue"))

    evaluator = LanguageEvaluator()

    # Generate and run tests
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating test suite...", total=None)
        test_cases = evaluator.generate_test_suite()
        progress.update(task, description=f"Generated {len(test_cases)} test cases")

        await asyncio.sleep(1)

        progress.update(task, description="Running evaluation...")
        metrics = await evaluator.run_evaluation()

        progress.update(task, description="Analyzing results...")
        await asyncio.sleep(0.5)

    # Display results
    report = evaluator.generate_report()
    console.print(report)

    # Generate improvement suggestions
    enhancer = AutonomousEnhancer()
    enhancements = enhancer.analyze_evaluation_results(evaluator.results)

    if enhancements:
        console.print(f"\n[bold green]💡 Identified {len(enhancements)} potential enhancements:[/bold green]")

        table = Table(title="Proposed Enhancements", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Impact", justify="right", style="green")
        table.add_column("Risk", style="yellow")

        for enhancement in enhancements[:5]:
            table.add_row(
                enhancement.type.value,
                enhancement.description[:50] + "...",
                f"{enhancement.impact_score:.1f}",
                enhancement.risk_level
            )

        console.print(table)

    return metrics, enhancements

async def run_multi_agent_collaboration():
    """Run multi-agent collaboration"""
    console.print(Panel.fit("🤝 [bold blue]Multi-Agent Collaboration[/bold blue]", border_style="blue"))

    orchestrator = MultiAgentOrchestrator()

    # Initialize agents
    console.print("\n[bold]Initializing Agent Team:[/bold]")
    orchestrator.initialize_agents()

    # Generate tasks
    tasks = orchestrator.generate_enhancement_tasks()
    console.print(f"\n[bold]Generated {len(tasks)} enhancement tasks[/bold]")

    # Show task distribution
    table = Table(title="Task Distribution", show_header=True, header_style="bold magenta")
    table.add_column("Task ID", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Priority", justify="center", style="yellow")
    table.add_column("Dependencies", style="dim")

    for task in tasks[:6]:
        table.add_row(
            task.id,
            task.description[:40] + "...",
            "⭐" * task.priority,
            ", ".join(task.dependencies) if task.dependencies else "-"
        )

    console.print(table)

    # Assign and execute tasks
    console.print("\n[bold]Assigning tasks to agents...[/bold]")
    orchestrator.assign_tasks()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Agents working on tasks...", total=None)
        results = await orchestrator.execute_tasks()
        progress.update(task, description=f"Completed {len(results)} tasks")

    # Consensus voting
    console.print("\n[bold]🗳️ Consensus Voting on Major Changes:[/bold]")

    proposals = [
        {"description": "Add WebGPU support for AI operations", "impact": "high"},
        {"description": "Implement lazy evaluation", "impact": "medium"},
        {"description": "Add type inference engine", "impact": "high"},
    ]

    approved = []
    for proposal in proposals:
        if await orchestrator.consensus_decision(proposal):
            approved.append(proposal)

    console.print(f"\n[green]✅ Approved {len(approved)}/{len(proposals)} proposals[/green]")

    return results, approved

async def run_continuous_improvement(cycles: int = 3):
    """Run continuous improvement cycles"""
    console.print(Panel.fit("♾️ [bold blue]Continuous Improvement System[/bold blue]", border_style="blue"))

    improvement_system = ContinuousImprovement()

    console.print(f"\n[bold]Running {cycles} improvement cycles...[/bold]\n")

    for cycle in range(1, cycles + 1):
        console.print(f"\n[bold cyan]═══ Cycle {cycle}/{cycles} ═══[/bold cyan]")

        # Run improvement cycle
        cycle_metrics = await improvement_system.run_improvement_cycle()

        # Show cycle results
        console.print(f"\n[bold]Cycle {cycle} Results:[/bold]")
        console.print(f"  • Success Rate: {cycle_metrics['success_rate']:.1f}%")
        console.print(f"  • Enhancements Applied: {cycle_metrics['enhancements_applied']}")
        console.print(f"  • Tests Passed: {cycle_metrics['tests_passed']}/{cycle_metrics['tests_total']}")

        if cycle < cycles:
            console.print(f"\n[dim]Waiting before next cycle...[/dim]")
            await asyncio.sleep(2)

    # Generate final report
    improvement_system.generate_final_report()

async def main():
    """Main entry point"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]🚀 PolyThLang Autonomous Enhancement System[/bold green]\n" +
        "[dim]Continuously improving the language through AI collaboration[/dim]",
        border_style="green"
    ))

    try:
        # Menu
        console.print("\n[bold]Select operation:[/bold]")
        console.print("1. Run Evaluation Cycle")
        console.print("2. Run Multi-Agent Collaboration")
        console.print("3. Run Continuous Improvement (3 cycles)")
        console.print("4. Run Complete Enhancement Pipeline")
        console.print("5. Exit")

        choice = console.input("\n[bold cyan]Enter choice (1-5): [/bold cyan]")

        if choice == "1":
            await run_evaluation_cycle()

        elif choice == "2":
            await run_multi_agent_collaboration()

        elif choice == "3":
            await run_continuous_improvement(3)

        elif choice == "4":
            # Run complete pipeline
            console.print("\n[bold green]Running Complete Enhancement Pipeline[/bold green]\n")

            # Phase 1: Evaluation
            console.print("[bold]Phase 1: Language Evaluation[/bold]")
            metrics, enhancements = await run_evaluation_cycle()

            console.print("\n" + "─" * 60 + "\n")

            # Phase 2: Multi-Agent Collaboration
            console.print("[bold]Phase 2: Multi-Agent Collaboration[/bold]")
            results, approved = await run_multi_agent_collaboration()

            console.print("\n" + "─" * 60 + "\n")

            # Phase 3: Continuous Improvement
            console.print("[bold]Phase 3: Continuous Improvement[/bold]")
            await run_continuous_improvement(2)

            # Final Summary
            console.print("\n" + "═" * 60)
            console.print(Panel.fit(
                "[bold green]✨ Enhancement Pipeline Complete![/bold green]\n\n" +
                f"• Initial Success Rate: {metrics.get('success_rate', 0):.1f}%\n" +
                f"• Enhancements Identified: {len(enhancements)}\n" +
                f"• Proposals Approved: {len(approved)}\n" +
                f"• Multi-Agent Tasks: {len(results)}\n\n" +
                "[dim]PolyThLang has been autonomously enhanced![/dim]",
                border_style="green"
            ))

        elif choice == "5":
            console.print("[dim]Exiting...[/dim]")
            return

        else:
            console.print("[red]Invalid choice![/red]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Enhancement system interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        if console.is_terminal:
            traceback.print_exc()

if __name__ == "__main__":
    # Check if required dependencies are installed
    try:
        import numpy
        import psutil
    except ImportError:
        console.print("[red]Missing required dependencies![/red]")
        console.print("Please install: pip install numpy psutil")
        sys.exit(1)

    asyncio.run(main())