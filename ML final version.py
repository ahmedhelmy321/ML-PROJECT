import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import tkinter as tk
import numpy as np


df = None
frames = {}
btn_font = ("Arial", 18)
column_var_list = []
model_var = None
result_label = None
columns_frame = None
plot_type_var = None


def init_main_frame(root):
    global model_var, result_label
    main_frame = ctk.CTkFrame(root)
    frames["main"] = main_frame
    main_frame.pack(fill="both", expand=True)

    
    container = ctk.CTkFrame(main_frame)
    container.place(relx=0.5, rely=0.5, anchor="center")

    
    upload_btn = ctk.CTkButton(container, text="1. Upload File", command=upload_file, font=btn_font, height=40, width=250, fg_color="#4CAF50", hover_color="#45A049")
    upload_btn.grid(row=0, column=0, padx=20, pady=15, sticky="ew")

    visualize_btn = ctk.CTkButton(container, text="2. Visualize Data", command=lambda: show_frame("visualize"), font=btn_font, height=40, width=250,fg_color="#2196F3", hover_color="#1E88E5")
    visualize_btn.grid(row=1, column=0, padx=20, pady=15, sticky="ew")

    preprocess_btn = ctk.CTkButton(container, text="3. Preprocess Data", command=preprocess_data, font=btn_font, height=40, width=250,fg_color="#FF9800", hover_color="#F57C00")
    preprocess_btn.grid(row=2, column=0, padx=20, pady=15, sticky="ew")

    
    model_frame = ctk.CTkFrame(container, fg_color="transparent", border_width=2, border_color="#B0BEC5")
    model_frame.grid(row=3, column=0, padx=20, pady=15, sticky="ew")
    model_label = ctk.CTkLabel(model_frame, text="4. Model Selection", font=btn_font)
    model_label.pack(pady=10)

    model_var = ctk.StringVar(value="Logistic Regression")  
    models = ["Logistic Regression", "Random Forest", "SVM", "Neural Network","KMeans Clustering"]
    
    for i, model in enumerate(models):
        ctk.CTkRadioButton(model_frame, text=model, variable=model_var, value=model, font=("Arial", 14)).pack(anchor='w', padx=3, pady=5)

        

    train_btn = ctk.CTkButton(container, text="5. Train & Evaluate Model", command=train_model, font=btn_font, height=40, width=250, fg_color="#D81B60", hover_color="#C2185B")
    train_btn.grid(row=4, column=0, padx=20, pady=15, sticky="ew")
   

    result_label = ctk.CTkLabel(container, text="", font=btn_font, wraplength=600)
    result_label.grid(row=5, column=0, padx=20, pady=15)


def init_visualize_frame(root):
    global columns_frame, plot_type_var
    visualize_frame = ctk.CTkFrame(root)
    frames["visualize"] = visualize_frame

    
    button_frame = ctk.CTkFrame(visualize_frame, fg_color="transparent")
    button_frame.pack(fill="x", pady=10)

    back_btn = ctk.CTkButton(button_frame, text="⬅️ Back", command=lambda: show_frame("main"), font=btn_font, height=40, width=150, fg_color="#757575", hover_color="#616161")
    back_btn.pack(side="left", padx=20)

    visualize_selected_btn = ctk.CTkButton(button_frame, text="📊 Visualize Plot", command=visualize_selected_plot, font=btn_font, height=40, width=150, fg_color="#4CAF50", hover_color="#45A049")
    visualize_selected_btn.pack(side="right", padx=20)

    columns_frame = ctk.CTkScrollableFrame(visualize_frame, width=300, height=300, fg_color="#ECEFF1")
    columns_frame.pack(pady=10, fill="x", padx=20)

    plot_type_var = ctk.StringVar(value="Box Plot")
    plot_type_menu = ctk.CTkOptionMenu(visualize_frame, values=["Box Plot", "Scatter Plot", "Line Plot"], variable=plot_type_var, font=btn_font, fg_color="#0288D1", button_color="#0288D1", button_hover_color="#0277BD")
    plot_type_menu.pack(pady=10)


def show_frame(name):   
    for f in frames.values():
        f.pack_forget() 
    frames[name].pack(fill="both", expand=True)
    if name == "visualize" and df is not None:
        populate_columns() 


def populate_columns():
    global column_var_list
    for widget in columns_frame.winfo_children():
        widget.destroy()
    column_var_list = []
    for col in df.select_dtypes(include=['number']).columns: 
        var = tk.BooleanVar()
        chk = ctk.CTkCheckBox(columns_frame, text=col, variable=var, font=("Arial", 14), text_color= '#000')
        chk.pack(anchor='w', padx=10, pady=5)
        column_var_list.append((col, var))


def upload_file():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            messagebox.showinfo("Success", f"File loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))


def preprocess_data():
    global df
    if df is None:
        messagebox.showerror("Error", "No dataset loaded")
        return
    if df.empty or df.shape[1] < 2:
        messagebox.showerror("Error", "Dataset is empty or has insufficient columns")
        return
    try:
        imputer = SimpleImputer(strategy="mean")   
        numeric_cols = df.select_dtypes(include=['number']).columns 
        if len(numeric_cols) > 0:
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols]) 
        
              
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
         
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))  
        scaler = StandardScaler()   
        df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])   
        messagebox.showinfo("Preprocessing", "Preprocessing completed!✅")
    except Exception as e:
        messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")


def train_model():
    global df
    if df is not None:
        try:
            model_name = model_var.get() 

        
            if model_name == "KMeans Clustering":
                X = df.select_dtypes(include=['number'])
                model = KMeans(n_clusters=3, random_state=42)
                clusters = model.fit_predict(X)
                df['Cluster'] = clusters
                result_label.configure(text="KMeans clustering applied.\n'Cluster' column added to dataset.")
                return 
            
            X = df.iloc[:, :-1]  
            y = df.iloc[:, -1]   
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_name == "Logistic Regression":
                model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "SVM":
                model = SVC(decision_function_shape='ovr')
            elif model_name == "Neural Network":
                classes_num = len(np.unique(y_train))
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(32, activation='relu'),
                ])
                if classes_num == 2:
                    model.add(Dense(1, activation= 'sigmoid'))
                    loss_fn = 'binary_crossentropy'
                else:
                    model.add(Dense(classes_num, activation='softmax'))
                    loss_fn = 'sparse_categorical_crossentropy'

                model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                loss, acc = model.evaluate(X_test, y_test, verbose=0)
                result_label.configure(text=f"Accuracy: {acc*100:.2f}%")
                return

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            result_label.configure(text=f"Accuracy: {acc*100:.2f}%\nConfusion Matrix:\n{cm}")

        except Exception as e:
            messagebox.showerror("Error", f"Training Failed: {str(e)}")



def visualize_selected_plot():
    global df
    if df is not None:
        selected_cols = [col for col, var in column_var_list if var.get()] 
        if not selected_cols:
            messagebox.showwarning("Warning", "Please select at least one column.")
            return

        plot_type = plot_type_var.get()  

        if plot_type == "Scatter Plot":
            if len(selected_cols) < 2:
                messagebox.showwarning("Warning", "Select at least two columns for a scatter plot.")
                return
            if len(selected_cols) % 2 != 0:
                messagebox.showwarning("Warning", "Please select an even number of columns for paired scatter plots.")
                return

            num_plots = len(selected_cols) // 2
            fig, axes = plt.subplots(nrows=num_plots, figsize=(6, 4 * num_plots))
            if num_plots == 1:
                axes = [axes]

            for i in range(num_plots):
                col_x = selected_cols[2 * i]
                col_y = selected_cols[2 * i + 1]

                if 'Cluster' in df.columns:
                    sns.scatterplot(data=df, x=col_x, y=col_y, hue='Cluster', palette='Set1', ax=axes[i])
                else:
                    sns.scatterplot(data=df, x=col_x, y=col_y, ax=axes[i])

                axes[i].set_title(f"{col_x} vs {col_y}")
                axes[i].set_xlabel(col_x)
                axes[i].set_ylabel(col_y)

            plt.tight_layout()
            plt.show()

        else:
            fig, axes = plt.subplots(nrows=len(selected_cols), figsize=(6, 4 * len(selected_cols)))
            if len(selected_cols) == 1:
                axes = [axes]

            for ax, col in zip(axes, selected_cols):
                try:
                    if plot_type == "Box Plot":
                        sns.boxplot(y=df[col], ax=ax)
                    elif plot_type == "Line Plot":
                        ax.plot(df[col])
                    ax.set_title(f"{plot_type} of {col}")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center')

            plt.tight_layout()
            plt.show()

                  


ctk.set_appearance_mode("white")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
win_width = 1000
win_height = 700
x = (screen_width - win_width) // 2
y = (screen_height - win_height) // 2
root.title("🧠 ML Project GUI")
root.geometry(f"{win_width}x{win_height}+{x}+{y}")
root.resizable(width= True, height= True)

init_main_frame(root)
init_visualize_frame(root)
show_frame("main")

root.mainloop()