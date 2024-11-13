import csv
from io import TextIOWrapper
from typing import Dict, List, Sequence

from langchain.document_loaders import CSVLoader
from langchain.docstore.document import Document
from langchain.document_loaders.helpers import detect_file_encodings


class CSVLoaderWithPreProcessing(CSVLoader):
    def __init__(
        self, 
        file_path: str, 
        source_column: str | None = None, 
        metadata_columns: Sequence[str] = (), 
        csv_args: Dict | None = None, 
        encoding: str | None = None, 
        autodetect_encoding: bool = False
    ):
        super().__init__(
            file_path, 
            source_column, 
            metadata_columns, 
            csv_args, 
            encoding, 
            autodetect_encoding
        )
    
    def load(self) -> List[Document]:
        """Load data into document objects."""

        docs = []
        try:
            with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
                docs = self.__read_file(csvfile)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    try:
                        with open(
                            self.file_path, newline="", encoding=encoding.encoding
                        ) as csvfile:
                            docs = self.__read_file(csvfile)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        return docs
    
    def __read_file(self, csvfile: TextIOWrapper) -> List[Document]:
        """Customize this function to add pre-processing stage."""
        
        docs = []
        
        csv_reader = csv.DictReader(csvfile, **self.csv_args)
        for i, row in enumerate(csv_reader):
            try:
                source = (
                    row[self.source_column]
                    if self.source_column is not None
                    else self.file_path
                )
            except KeyError:
                raise ValueError(
                    f"Source column '{self.source_column}' not found in CSV file."
                )
            row = self.__preprocess_row(row)
            content = "\n".join(
                f"{k.strip()}: {v.strip()}"
                for k, v in row.items()
                if k not in self.metadata_columns
            )
            metadata = {"source": source, "row": i}
            
            for col in self.metadata_columns:
                try:
                    metadata[col] = row[col]
                except KeyError:
                    raise ValueError(f"Metadata column '{col}' not found in CSV file.")
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)

        return docs

    def __preprocess_row(
        self, 
        row_dict: Dict[str, str]
    ) -> Dict[str, str]:
        """pre-process each row dictionary."""
        
        preprocessed_row = {}
        
        for k, v in row_dict.items():
            if v.strip() == '': continue
            
            preprocessed_v = v.replace("\r\n", "")
            preprocessed_row[k] = preprocessed_v
                
        return preprocessed_row