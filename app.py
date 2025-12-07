"""
Hola!! 

Estos son los pasos para correr esta app: 
1. Crea una carpeta. 
2. Guarda en esa carpeta este archivo bajo el nombre app.py
3. Guarda en esa carpeta el archivo del modelo (solamente funciona con el modelo final) bajo el nombre de model.pth
    --> Yo recomiendo usar el de la época 4 (la de mejor performance). 
4. Abre la carpeta en positron o vs studio y corre el script de app.py
5. Se abrirá en la terminal un link que te lleva a una pagina en tu local host. 
    --> Puedes escoger si abrirla dentro del mismo IDE, pero a mí me gusta más la experiencia desde tu default browser. 
    --> Probablemente no te corra la primera vez, pero es cosa de instalar las liberías con "pip install" debería funcionar sin problema :) 
    
Y listo! 

PD. Si gustas que te comparta la carpeta comprimida lista para correr dime en un correo y con gusto lo hago... solo que el archivo del modelo fue muy pesado para github :/ 
"""

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import os
from uuid import uuid4
from shiny import App, ui, render


class gatoperroCNN_EfficientNet(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=True):
        super().__init__()

        self.backbone = models.efficientnet_b0(pretrained=use_pretrained)

        for param in self.backbone.features[:5].parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = gatoperroCNN_EfficientNet(num_classes=2, use_pretrained=False).to(device)

checkpoint = torch.load("model.pth", map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


class_names = checkpoint.get("class_names", ["Cat", "Dog"])


infer_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def predict_pil_image(pil_img: Image.Image):

    img_rgb = pil_img.convert("RGB")
    x = infer_transform(img_rgb).unsqueeze(0).to(device)  

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = probs.argmax()
    pred_class = class_names[pred_idx]
    return pred_class, probs


app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.link(
            rel="stylesheet",
            href=(
                "https://fonts.googleapis.com/css2?"
                "family=Poppins:wght@300;400;600;700&display=swap"
            ),
        ),
        ui.tags.style(
            """
            body {
                background-color: #000000;
                color: #ffffff;
                font-family: 'Poppins', sans-serif;
            }

            .app-container {
                max-width: 900px;
                margin: 0 auto;
                padding: 30px 15px 50px 15px;
            }

            .app-title {
                text-align: center;
                font-weight: 700;
                letter-spacing: 0.03em;
                margin-bottom: 8px;
                color: #ffffff;
            }

            .app-subtitle {
                text-align: center;
                font-weight: 300;
                font-size: 1rem;
                margin-bottom: 30px;
                opacity: 0.85;
                color: #ffffff;
            }

            .sidebar {
                background-color: #111111 !important;
                border-radius: 18px;
                padding: 20px 18px;
                box-shadow: 0 0 25px rgba(255,255,255,0.08);
            }

            .shiny-input-container {
                text-align: center;
            }

            .shiny-input-container > label {
                color: #ffffff;
                font-weight: 500;
                margin-bottom: 8px;
            }

            .sidebar p {
                font-size: 0.85rem;
                opacity: 0.7;
                margin-top: 8px;
                color: #ffffff;
            }

            .app-main-card {
                background-color: #111111;
                border-radius: 18px;
                padding: 25px 25px 30px 25px;
                box-shadow: 0 0 25px rgba(255,255,255,0.08);
                margin-left: 15px;
            }

            .preview-title,
            .prediction-title {
                text-align: center;
                margin-top: 5px;
                margin-bottom: 15px;
                font-weight: 600;
                color: #ffffff;
            }

            .image-frame {
                max-width: 520px;
                height: 380px;          
                margin: 0 auto;
                border-radius: 16px;
                overflow: hidden;      
                box-shadow: 0 0 18px rgba(255,255,255,0.1);
                background: #000000;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            #preview_img {
                width: auto;
                height: auto;
                object-fit: auto;
            }
            #prediction_text {
                text-align: left;
                font-size: 0.95rem;
                font-weight: 400;
                margin-top: 10px;
                padding: 12px 18px;
                border-radius: 16px;
                background: #000000;
                color: #ffffff;
                display: block;
                max-width: 100%;
                white-space: pre-wrap;
                word-break: break-word;
                margin-left: auto;
                margin-right: auto;
                border: 1px solid rgba(255,255,255,0.3);
            }

            .prediction-wrapper {
                text-align: center;
                margin-top: 10px;
            }

            @media (max-width: 767px) {
                .app-main-card {
                    margin-left: 0;
                    margin-top: 20px;
                }
            }
            """
        ),
    ),

    ui.div(
        ui.h2(
            "Clasificador de imágenes en gatos vs. perros",
            class_="app-title"
        ),
        ui.p(
            "Sube una foto para ver si el algoritmo predice que es un perro o un gato :)",
            class_="app-subtitle",
        ),

        ui.layout_columns(
            ui.div(
                ui.h4("Sube tu imagen", class_="preview-title"),
                ui.input_file(
                    "image_file",
                    "Sube una imagen",
                    multiple=False,
                    accept=["image/png", "image/jpeg", "image/jpg"],
                ),
                ui.p(
                    "Por favor, solo sube imágenes con formato .png, .jpg ó .jpeg"
                ),
                class_="sidebar",   
            ),

            ui.div(
                ui.h4("Tu imagen:", class_="preview-title"),
                ui.div(
                    ui.output_image("preview_img"),
                    class_="image-frame",
                ),
                ui.br(),
                ui.h4("Predicción del modelo:", class_="prediction-title"),
                ui.div(
                    ui.output_text_verbatim("prediction_text"),
                    class_="prediction-wrapper",
                ),
                class_="app-main-card",
            ),
            col_widths=(5, 7),
        ),

        class_="app-container",
    ),
)

def server(input, output, session):
    @render.image
    def preview_img():
        fileinfo_list = input.image_file()
        if not fileinfo_list:
            return None

        fileinfo = fileinfo_list[0]
        img_path = fileinfo["datapath"]

        img = Image.open(img_path).convert("RGB")
        target_w, target_h = 520, 380
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(img_resized, (offset_x, offset_y))

        out_dir = "tmp_preview"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"preview_{uuid4().hex}.png")
        canvas.save(out_path)

        return {
            "src": out_path,
            "alt": "Imagen subida",
            "width": "100%",  
        }


    @render.text
    def prediction_text():
        fileinfo_list = input.image_file()
        if not fileinfo_list:
            return "Sube una imagen para hacer la predicción."

        fileinfo = fileinfo_list[0]
        img_path = fileinfo["datapath"]

        img = Image.open(img_path).convert("RGB")
        pred_class, probs = predict_pil_image(img)

        msg = (
            f"Predicción: {pred_class}\n\n"
            f"Probabilidades:\n"
            f"  {class_names[0]}: {probs[0]:.4f}\n"
            f"  {class_names[1]}: {probs[1]:.4f}"
        )
        return msg


app = App(app_ui, server)

