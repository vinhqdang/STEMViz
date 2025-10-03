#!/usr/bin/env python3
"""
Gradio web app for STEM Animation Generator
Simple UI: Input concept -> Progress bar -> Video player
"""

import gradio as gr
from pipeline import Pipeline
from pathlib import Path

pipeline = Pipeline()

def generate_animation(concept: str, progress=gr.Progress()):
    """
    Main generation function called by Gradio
    
    Args:
        concept: User input STEM concept
        progress: Gradio progress tracker
        
    Returns:
        Video file path or error message
    """
    if not concept or concept.strip() == "":
        return None
    
    def update_progress(message: str, percentage: float):
        progress(percentage, desc=message)
    
    result = pipeline.run(concept, progress_callback=update_progress)

    if result["status"] == "success" and result.get("video_result"):
        video_path = result["video_result"].get("output_path")
        if video_path and Path(video_path).exists():
            return video_path
        else:
            error_msg = result.get("error_message", "Video file not found")
            gr.Warning(f"Generation completed but video file not available: {error_msg}")
            return None
    else:
        error_msg = result.get("error_message", "Unknown error occurred")
        gr.Warning(f"Generation failed: {error_msg}")
        return None

with gr.Blocks(title="STEMViz") as demo:
    gr.Markdown("# STEMViz")
    gr.Markdown("Transform STEM concepts into narrated educational animations")
    
    with gr.Row():
        with gr.Column():
            concept_input = gr.Textbox(
                label="Enter STEM Concept",
                placeholder="e.g., Explain Bubble Sort, Bayes' Theorem, Gradient Descent...",
                lines=2
            )
            generate_btn = gr.Button("Generate Animation", variant="primary")
        
    with gr.Row():
        video_output = gr.Video(
            label="Generated Animation",
            autoplay=True
        )
    
    gr.Examples(
        examples=[
            ["Explain Bubble Sort"],
            ["Explain Bayes' Theorem"],
            ["Explain Gradient Descent"]
        ],
        inputs=concept_input
    )
    
    generate_btn.click(
        fn=generate_animation,
        inputs=concept_input,
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
