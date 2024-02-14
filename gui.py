import tkinter as tk
from tkinter import ttk
import os
import threading
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import libtmux

import diffusion


WINDOW_SIZE = '1000x600'
CANVAS_SIZE = 200

def get_selected_model():
    model = None
    for i in models_list.curselection():
        model = models_list.get(i)
        break
    return model

def train():
    model = get_selected_model()
    if model:
        server = libtmux.Server()
        session = server.new_session(session_name=f'training_{model}', kill_session=True, attach=False)
        pane = session.windows[0].attached_pane
        pane.send_keys(f'python diffusion.py {model} {steps_value.get()} {lr_value.get()} {batch_value.get()} {dfts_value.get()} {int(paired_dataset.get())}')

        toggle_follow(force_sunken=True)
        tk.messagebox.showinfo(title='Info', message=f'Training for {model} has started.\nLog following is enabled.')
    else:
        tk.messagebox.showerror(title='Error', message='No model selected.')

def follow():
    server = libtmux.Server()
    while follow_button.config('relief')[-1] == 'sunken':
        model = get_selected_model()
        if model:
            try:
                session = server.sessions.get(session_name=f'training_{model}')
                pane = session.windows[0].attached_pane
                train_label.config(text='\n'.join(pane.cmd('capture-pane', '-p').stdout[-3:]))
            except libtmux._internal.query_list.ObjectDoesNotExist:
                train_label.config(text='No training active.')
        else:
            train_label.config(text='No model selected.')
        time.sleep(0.2)
    on_model_change()

def toggle_follow(force_sunken=False):
    if follow_button.config('relief')[-1] == 'sunken' and not force_sunken:
        follow_button.config(relief="raised")
    else:
        follow_button.config(relief="sunken")
        p = threading.Thread(target=follow, args=())
        p.start()

def generate():
    model = get_selected_model()
    if model:
        generate_button.config(state=tk.DISABLED)
        p = threading.Thread(target=diffusion.sample, args=(model, generated_images, CANVAS_SIZE, on_progress_change))
        p.start()
    else:
        tk.messagebox.showerror(title='Error', message='No model selected.')

def on_model_change(event=None):
    model = get_selected_model()
    if model:
        progress.config(text='')
        generate_button.config(state=tk.NORMAL)
        train_label.config(text=f'Set hyperparameters for {model}.')
        train_button.config(state=tk.NORMAL)
        p = threading.Thread(target=diffusion.set_metadata, args=(model, metadata))
        p.start()
        p = threading.Thread(target=diffusion.plot_losses, args=(model, graph_canvas, axis))
        p.start()
        p = threading.Thread(target=diffusion.set_dataset_preview, args=(model, images, CANVAS_SIZE, paired_dataset.get()))
        p.start()
        if not diffusion.is_model_trained(model):
            progress.config(text='Model not yet trained.')
            generate_button.config(state=tk.DISABLED)
    else:
        progress.config(text='No model selected.')
        generate_button.config(state=tk.DISABLED)
        train_label.config(text='No model selected.')
        train_button.config(state=tk.DISABLED)
        metadata.config(text='Welcome!\nSelect a model from the sidebar.')
        axis.clear()
        axis.set_ylabel('Loss')
        axis.set_xlabel('Epoch')
        graph_canvas.draw()

def on_progress_change(i, n):
    progress.config(text=f'Generating... {int((n-i)/n*100)}%')
    if (i == 0):
        progress.config(text='')
        generate_button.config(state=tk.NORMAL)

if __name__ == "__main__":

    # Root window
    window = tk.Tk()
    window.title("Diffusion Playground")
    window.geometry(WINDOW_SIZE)
    window.bind('<Control-q>', quit)

    # Left sidebar
    sidebar = tk.Frame(window)

    models_title = tk.Label(sidebar, text='Models')
    models_list = tk.Listbox(sidebar)
    models_list.bind('<<ListboxSelect>>', on_model_change)
    def clear_selection(event):
        models_list.selection_clear(0, tk.END)
        on_model_change(event)
    window.bind('<Control-w>', clear_selection)
    for filename in os.listdir('data'):
        models_list.insert(tk.END, filename)
    
    paired_dataset = tk.BooleanVar(value=True)
    paired_check = tk.Checkbutton(sidebar, text='Paired dataset',variable=paired_dataset, onvalue=1, offvalue=0, command=on_model_change)

    sidebar.pack(side=tk.LEFT)
    models_title.pack()
    models_list.pack()
    paired_check.pack()

    # Notebook panel (tabs)
    notebook_style = ttk.Style(window)
    notebook_style.configure('bottom.TNotebook', tabposition='sw')
    notebook = ttk.Notebook(window, style='bottom.TNotebook')
    notebook.pack(expand=True, fill='both')

    metadata_frame = tk.Frame(notebook)
    dataset_frame = tk.Frame(notebook)
    train_frame = tk.Frame(notebook)
    generate_frame = tk.Frame(notebook)

    metadata_frame.pack(fill='both', expand=True)
    dataset_frame.pack(fill='both', expand=True)
    train_frame.pack(fill='both', expand=True)
    generate_frame.pack(fill='both', expand=True)

    notebook.add(metadata_frame, text='Metadata')
    notebook.add(dataset_frame, text='Dataset')
    notebook.add(train_frame, text='Train')
    notebook.add(generate_frame, text='Generate')

    # Metadata tab
    metadata = tk.Label(metadata_frame)
    metadata.pack(fill=tk.BOTH, expand=True)

    fig = Figure(figsize=(5,5), dpi=100)
    axis = fig.add_subplot(1, 1, 1)
    fig.set_facecolor('lightgray')
    axis.set_facecolor('lightgray')

    graph_canvas = FigureCanvasTkAgg(fig, metadata_frame)
    graph_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Dataset tab
    scrollbar = tk.Scrollbar(dataset_frame)
    scrollbar.pack(side = tk.RIGHT, fill=tk.Y)

    image_canvas = tk.Canvas(dataset_frame, height=CANVAS_SIZE*4, width=CANVAS_SIZE*2)    
    image_grid = tk.Frame(image_canvas, height=CANVAS_SIZE*4, width=CANVAS_SIZE*2)
    image_grid.pack()
    image_canvas.create_window(0, 0, window=image_grid, anchor="nw")
    image_canvas.pack()
    image_canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.config(command = image_canvas.yview)

    images = [tk.Canvas(image_grid, height=CANVAS_SIZE, width=CANVAS_SIZE) for _ in range(8)]
    images[0].grid(row=0, column=0)
    images[1].grid(row=0, column=1)
    images[2].grid(row=1, column=0)
    images[3].grid(row=1, column=1)
    images[4].grid(row=2, column=0)
    images[5].grid(row=2, column=1)
    images[6].grid(row=3, column=0)
    images[7].grid(row=3, column=1)

    image_canvas.configure(scrollregion=image_canvas.bbox("all"))

    # Train tab
    follow_button = tk.Button(master = train_frame,
                              command = toggle_follow,
                              height = 2,
                              width = 10,
                              text = 'Follow logs')
    follow_button.pack(side=tk.BOTTOM, pady=10)

    train_label = tk.Label(train_frame)
    train_label.pack(side=tk.BOTTOM)

    setup = tk.Frame(train_frame)
    steps_label = tk.Label(setup, text='Training steps')
    steps_value = tk.StringVar(value='2e5')
    steps_input = tk.Entry(setup, textvariable=steps_value)
    lr_label = tk.Label(setup, text='Learning rate')
    lr_value = tk.StringVar(value='1e-4')
    lr_input = tk.Entry(setup, textvariable=lr_value)
    batch_label = tk.Label(setup, text='Batch size')
    batch_value = tk.StringVar(value='16')
    batch_input = tk.Entry(setup, textvariable=batch_value)
    dfts_label = tk.Label(setup, text='Diffusion timesteps')
    dfts_value = tk.StringVar(value='1000')
    dfts_input = tk.Entry(setup, textvariable=dfts_value)

    train_button = tk.Button(master = setup,
                             command = train,
                             height = 2,
                             width = 10,
                             text = 'Start training')
    
    setup.pack(expand=True)
    steps_label.pack()
    steps_input.pack()
    lr_label.pack()
    lr_input.pack()
    batch_label.pack()
    batch_input.pack()
    dfts_label.pack()
    dfts_input.pack()
    train_button.pack(pady=10)

    # Generate tab
    generate_button = tk.Button(master = generate_frame,
                     command = generate,
                     height = 2,
                     width = 20,
                     text = "Generate new images")
    generate_button.pack(side=tk.BOTTOM, pady=10)

    generate_grid = tk.Frame(generate_frame)
    generate_grid.pack(expand=True)
    
    progress = tk.Label(generate_frame)
    progress.pack(side=tk.BOTTOM)

    generated_images = [tk.Canvas(generate_grid, height=CANVAS_SIZE, width=CANVAS_SIZE) for i in range(4)]
    generated_images[0].grid(row=0, column=0)
    generated_images[1].grid(row=0, column=1)
    generated_images[2].grid(row=1, column=0)
    generated_images[3].grid(row=1, column=1)

    on_model_change()

    window.mainloop()
