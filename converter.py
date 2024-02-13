from haystack.pipelines import Pipeline
from haystack.nodes import TextConverter, FileTypeClassifier, PDFToTextConverter, MarkdownConverter, DocxToTextConverter, PreProcessor


class Converter():

    def __init__(self):
        self.p = Pipeline()
        self.file_type_classifier = FileTypeClassifier()

        self.text_converter = TextConverter()
        self.pdf_converter = PDFToTextConverter()
        self.md_converter = MarkdownConverter()
        self.docx_converter = DocxToTextConverter()
        self.preprocessor = PreProcessor()
        self.p.add_node(component=self.file_type_classifier, name="FileTypeClassifier", inputs=["File"])

        self.p.add_node(component=self.text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
        self.p.add_node(component=self.pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"])
        self.p.add_node(component=self.md_converter, name="MarkdownConverter", inputs=["FileTypeClassifier.output_3"])
        self.p.add_node(component=self.docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])

        self.p.add_node(
            component=self.preprocessor,
            name="Preprocessor",
            inputs=["TextConverter", "PdfConverter", "MarkdownConverter", "DocxConverter"],
        )

    def run_converter(self, input):
        return self.p.run(file_paths=[input])




