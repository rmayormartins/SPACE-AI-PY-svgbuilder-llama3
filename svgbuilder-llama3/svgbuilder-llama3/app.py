import os
from groq import Groq
import gradio as gr
import aiofiles

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

system_prompt = {
    "role": "system",
    "content": "You are a useful assistant that generates and refines SVG diagrams. Ensure the SVG code is valid and starts with <svg>."
}

previous_svg = ""

async def generate_diagram_llama(description, option, existing_svg=None):
    global previous_svg
    messages = [system_prompt]

    if option == "Refinar anterior" and previous_svg:
        messages.append({"role": "user", "content": f"Here is the existing SVG diagram: {previous_svg}"})
        messages.append({"role": "user", "content": f"Refine the SVG diagram based on the following description: {description}"})
    elif option == "Refinar existente" and existing_svg:
        messages.append({"role": "user", "content": f"Here is the existing SVG diagram: {existing_svg}"})
        messages.append({"role": "user", "content": f"Refine the SVG diagram based on the following description: {description}"})
    else:
        messages.append({"role": "user", "content": f"Generate an SVG diagram based on the following description: {description}"})

    response_content = ''
    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=1024,
        temperature=1.3,
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += chunk.choices[0].delta.content

    print("Resposta da API:", response_content)

    if "```" in response_content:
        svg_content = response_content.split("```")[1].strip()
    else:
        svg_content = response_content.strip()

    if not svg_content.startswith("<svg"):
        svg_content = "<svg width='100' height='100' xmlns='http://www.w3.org/2000/svg'><text x='10' y='20'>Invalid SVG</text></svg>"

    previous_svg = svg_content
    print("SVG Anterior Armazenado:", previous_svg)
    return svg_content

async def create_svg_file(svg_code):
    filename = "generated_diagram.svg"
    async with aiofiles.open(filename, "w") as file:
        await file.write(svg_code)
    return filename

async def generate_and_display_diagram(description, option, existing_svg=None):
    svg_code = await generate_diagram_llama(description, option, existing_svg)
    svg_file = await create_svg_file(svg_code)
    return svg_code, svg_file

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    description_input = gr.Textbox(label="Description")
    option_input = gr.Radio(choices=["Gerar novo", "Refinar anterior", "Refinar existente"], label="Opção")
    existing_svg_input = gr.File(label="Upload Existing SVG (Optional)", visible=False)
    svg_display = gr.HTML()
    output_file = gr.File(label="Generated SVG")
    submit_button = gr.Button("Generate")

    async def update_output(description, option, existing_svg_file):
        existing_svg = None
        if option == "Refinar existente" and existing_svg_file:
            async with aiofiles.open(existing_svg_file.name, "r") as file:
                existing_svg = await file.read()
        svg_code, svg_file = await generate_and_display_diagram(description, option, existing_svg)
        return svg_code, svg_file

    def toggle_file_input(option):
        if option == "Refinar existente":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)

    option_input.change(toggle_file_input, inputs=option_input, outputs=existing_svg_input)
    submit_button.click(update_output, inputs=[description_input, option_input, existing_svg_input], outputs=[svg_display, output_file])

demo.queue()
demo.launch(share=True)
