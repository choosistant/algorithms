import gradio as gr

from src.api_server import predict


text_input = gr.Textbox(lines=8,placeholder="Enter the review text here...",label="Review Text")
new_title = gr.Textbox(label="Detected pro/con")
iface = gr.Interface(
    predict, [text_input],
    new_title,
    description ="Checkout some amazon reviews eg at [http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz)",
    allow_flagging="manual",
    flagging_options = ['Incorrect','Offensive', 'Other'],
    examples = [['The top side came ripped off'],
                ['The gear is child friendly']],
    )

iface.launch()