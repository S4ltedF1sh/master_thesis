"""
Generate the S2ORC-ITG-Subset.
"""
import glob
import gzip
import io
import json
import logging
import multiprocessing
import os
from typing import Dict, Any

import tqdm
from intertext_graph.itgraph import IntertextDocument, Node, Edge, Etype, SpanNode
from intertext_graph.parsers.itparser import IntertextParser

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

NTYPE_PARAGRAPH = "p"
NTYPE_TITLE = "title"
NTYPE_ARTICLE_TITLE = "article-title"
NTYPE_ABSTRACT = "abstract"

NTYPE_CITE_SPAN = "cite-span"
NTYPE_REF_SPAN = "ref-span"

NTYPE_BIB_ENTRIES_SECTION = "bib-entries-section"
NTYPE_REF_ENTRIES_SECTION = "ref-entries-section"

NTYPE_BIB_ENTRY = "bib-entry"
NTYPE_REF_ENTRY = "ref-entry"


class S2ORCParser(IntertextParser):
    """
    Parser to transform a S2ORC document into an IntertextDocument.

    Description of the S2ORC data model: https://github.com/allenai/s2orc
    """

    def __init__(self, metadata_path: str, pdf_parse_path: str, paper_id: str):
        """
        Initialize the S2ORCParser for a particular paper.

        Note that the metadata and PDF parse JSONL files contain the metadata and PDF parses of multiple documents.

        :param metadata_path: path of the JSONL file that contains (among others) the metadata of the document
        :param pdf_parse_path: path of the JSONL file that contains (among others) the PDF parse of the document
        :param paper_id: identifier of the particular paper
        """
        super(S2ORCParser, self).__init__(metadata_path)
        self._metadata_path: str = metadata_path
        self._pdf_parse_path: str = pdf_parse_path
        self._paper_id: str = paper_id

    def __call__(self, metadata_jsonl=None, pdf_parse_jsonl=None) -> IntertextDocument:
        """
        Parse the S2ORC document into an IntertextDocument.

        If the metadata and PDF parse are provided through the method arguments, the method will skip reading the
        corresponding files again.

        :param metadata_jsonl: content of the JSONL file that contains (among others) the metadata of the document
        :param pdf_parse_jsonl: content of the JSONL file that contains (among others) the PDF parse of the document
        :return: the IntertextDocument
        """
        # load the file contents if they are not provided through the method arguments
        if metadata_jsonl is None:
            metadata_jsonl = []
            with open(self._metadata_path, "r", encoding="utf-8") as file:
                for line in file:
                    metadata_jsonl.append(json.loads(line))

        if pdf_parse_jsonl is None:
            pdf_parse_jsonl = []
            with open(self._pdf_parse_path, "r", encoding="utf-8") as file:
                for line in file:
                    pdf_parse_jsonl.append(json.loads(line))

        # find correct metadata and PDF parse based on paper id
        metadata_json = None
        for metadata in metadata_jsonl:
            if metadata["paper_id"] == self._paper_id:
                metadata_json = metadata
                break
        if metadata_json is None:
            logger.error(f"Could not find the metadata of the document with the paper id {self._paper_id}!")
            assert False, f"Could not find the metadata of the document with the paper id {self._paper_id}!"

        pdf_parse_json = None
        for pdf_parse in pdf_parse_jsonl:
            if pdf_parse["paper_id"] == self._paper_id:
                pdf_parse_json = pdf_parse
                break
        if pdf_parse_json is None:
            logger.error(f"Could not find the PDF parse of the document with the paper id {self._paper_id}!")
            assert False, f"Could not find the PDF parse of the document with the paper id {self._paper_id}!"

        # make sure that all required elements exist
        assert metadata_json["has_pdf_parse"]
        assert metadata_json["has_pdf_parsed_abstract"]
        assert metadata_json["has_pdf_parsed_body_text"]
        assert metadata_json["has_pdf_parsed_bib_entries"]
        assert metadata_json["has_pdf_parsed_ref_entries"]

        return self._parse_document(metadata_json, pdf_parse_json)

    @classmethod
    def _batch_func(cls, path: Any) -> Any:
        raise NotImplementedError  # TODO: implement this

    def _parse_document(self, metadata_json: Dict[str, Any], pdf_parse_json: Dict[str, Any]) -> IntertextDocument:
        """
        Parse the given S2ORC Document.

        The document comprises the metadata and the PDF parse.

        :param metadata_json: metadata of the document
        :param pdf_parse_json: PDF parse of the document
        :return:
        """

        # create intertext document
        prefix = metadata_json["paper_id"]
        metadata = self._create_document_metadata(metadata_json, pdf_parse_json)

        intertext_document = IntertextDocument(
            nodes=[],
            edges=[],
            prefix=prefix,
            meta=metadata
        )

        # create article title as root
        article_title_node = Node(
            content=metadata_json["title"],
            ntype=NTYPE_ARTICLE_TITLE
        )
        intertext_document.add_node(article_title_node)

        # parse reference entries
        ref_entries_section_node = Node(
            content="Reference Entries",  # magic string
            ntype=NTYPE_REF_ENTRIES_SECTION
        )
        intertext_document.add_node(ref_entries_section_node)

        ref_entries_section_parent_edge = Edge(
            src_node=article_title_node,
            tgt_node=ref_entries_section_node,
            etype=Etype.PARENT
        )
        intertext_document.add_edge(ref_entries_section_parent_edge)

        # the predecessor of this node is the final paragraph of the body_text, so the next edge must be defined at the
        # end of the method

        ref_entry_nodes = {}
        pred_node = ref_entries_section_node
        for ref_entry_id, ref_entry_json in pdf_parse_json["ref_entries"].items():
            pred_node = self._parse_ref_entry(ref_entry_id, ref_entry_json,
                                              intertext_document, ref_entries_section_node, pred_node,
                                              ref_entry_nodes)

        # parse bibliography entries
        bib_entries_section_node = Node(
            content="Bibliography Entries",  # magic string
            ntype=NTYPE_BIB_ENTRIES_SECTION
        )
        intertext_document.add_node(bib_entries_section_node)

        bib_entries_section_parent_edge = Edge(
            src_node=article_title_node,
            tgt_node=bib_entries_section_node,
            etype=Etype.PARENT
        )
        intertext_document.add_edge(bib_entries_section_parent_edge)

        bib_entries_section_next_edge = Edge(
            src_node=pred_node,
            tgt_node=bib_entries_section_node,
            etype=Etype.NEXT
        )
        intertext_document.add_edge(bib_entries_section_next_edge)

        bib_entry_nodes = {}
        pred_node = bib_entries_section_node
        for bib_entry_id, bib_entry_json in pdf_parse_json["bib_entries"].items():
            pred_node = self._parse_bib_entry(bib_entry_id, bib_entry_json,
                                              intertext_document, bib_entries_section_node, pred_node,
                                              bib_entry_nodes)

        # parse abstract
        if pdf_parse_json["abstract"] != []:
            abstract_section_node = Node(
                content="Abstract",  # magic string
                ntype=NTYPE_ABSTRACT
            )
            intertext_document.add_node(abstract_section_node)

            abstract_section_parent_edge = Edge(
                src_node=article_title_node,
                tgt_node=abstract_section_node,
                etype=Etype.PARENT
            )
            intertext_document.add_edge(abstract_section_parent_edge)

            abstract_section_next_edge = Edge(
                src_node=article_title_node,
                tgt_node=abstract_section_node,
                etype=Etype.NEXT
            )
            intertext_document.add_edge(abstract_section_next_edge)

            pred_node = abstract_section_node
            for paragraph_json in pdf_parse_json["abstract"]:
                assert paragraph_json["section"] == "Abstract", f"Title '{paragraph_json['section']}' != 'Abstract'!"
                pred_node = self._parse_paragraph(paragraph_json,
                                                  intertext_document, abstract_section_node, pred_node,
                                                  bib_entry_nodes, ref_entry_nodes)

        # parse body text
        if pdf_parse_json["body_text"] != []:
            current_section_node = None
            current_section_title = ""

            for paragraph_json in pdf_parse_json["body_text"]:
                if current_section_node is None or paragraph_json["section"] != current_section_title:
                    # create a new section with the new section title
                    current_section_title = paragraph_json["section"]
                    current_section_node = Node(
                        content=paragraph_json["section"],
                        ntype=NTYPE_TITLE
                    )
                    intertext_document.add_node(current_section_node)

                    current_section_parent_edge = Edge(
                        src_node=article_title_node,
                        tgt_node=current_section_node,
                        etype=Etype.PARENT
                    )
                    intertext_document.add_edge(current_section_parent_edge)

                    current_section_next_edge = Edge(
                        src_node=pred_node,
                        tgt_node=current_section_node,
                        etype=Etype.NEXT
                    )
                    intertext_document.add_edge(current_section_next_edge)
                    pred_node = current_section_node

                # parse the paragraph as part of the current section
                pred_node = self._parse_paragraph(paragraph_json,
                                                  intertext_document, current_section_node, pred_node,
                                                  bib_entry_nodes, ref_entry_nodes)

        # attach the reference entries and the bibliography entries to the end of the body text and abstract
        ref_entries_section_next_edge = Edge(
            src_node=pred_node,
            tgt_node=ref_entries_section_node,
            etype=Etype.NEXT
        )
        intertext_document.add_edge(ref_entries_section_next_edge)

        return intertext_document

    def _create_document_metadata(self, metadata_json: Dict[str, Any],
                                  pdf_parse_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the metadata of the intertext document from the S2ORC metadata and PDF parse.

        :param metadata_json: metadata of the document
        :param pdf_parse_json: PDF parse of the document
        :return: intertext document metadata
        """
        return {
            # identifier fields
            "paper_id": metadata_json["paper_id"],  # str
            "arxiv_id": metadata_json["arxiv_id"],  # str
            "acl_id": metadata_json["acl_id"],  # str
            "pmc_id": metadata_json["pmc_id"],  # str
            "pubmed_id": metadata_json["pubmed_id"],  # str
            "mag_id": metadata_json["mag_id"],  # str
            "doi": metadata_json["doi"],  # str

            # S2ORC metadata
            "title": metadata_json["title"],  # str
            "authors": metadata_json["authors"],  # [{"first": str, "middle": [str], "last": str, "suffix": str}]
            "venue": metadata_json["venue"],  # str
            "journal": metadata_json["journal"],  # str
            "year": metadata_json["year"],  # int
            "abstract": metadata_json["abstract"],  # str
            "inbound_citations": metadata_json["inbound_citations"],  # List[str] (paper_ids)
            "outbound_citations": metadata_json["outbound_citations"],  # List[str] (paper_ids)
            "has_inbound_citations": metadata_json["has_inbound_citations"],  # bool
            "has_outbound_citations": metadata_json["has_outbound_citations"],  # bool

            "mag_field_of_study": metadata_json["mag_field_of_study"],  # List[str]

            # PDF parse-related metadata (fields are missing if metadata["has_pdf_parse"] is false)
            "has_pdf_parse": metadata_json["has_pdf_parse"],  # bool
            "has_pdf_parsed_abstract": metadata_json["has_pdf_parse"] and metadata_json["has_pdf_parsed_abstract"],
            # bool
            "has_pdf_parsed_body_text": metadata_json["has_pdf_parse"] and metadata_json["has_pdf_parsed_body_text"],
            # bool
            "has_pdf_parsed_bib_entries": metadata_json["has_pdf_parse"] and metadata_json[
                "has_pdf_parsed_bib_entries"],  # bool
            "has_pdf_parsed_ref_entries": metadata_json["has_pdf_parse"] and metadata_json["has_pdf_parsed_ref_entries"]
            # bool
        }

    def _parse_paragraph(self, paragraph_json: Dict[str, Any],
                         intertext_document: IntertextDocument, section_node: Node, pred_node: Node,
                         bib_entry_nodes: Dict[str, Node], ref_entry_nodes: Dict[str, Node]) -> Node:
        """
        Parse the given S2ORC paragraph.

        {"section": str, "text": str, "cite_spans": [...], "ref_spans": [...]}

        :param paragraph_json: S2ORC paragraph
        :param intertext_document: intertext document object
        :param section_node: paragraph's section node
        :param pred_node: paragraph's predecessor node
        :param bib_entry_nodes: mapping from bibliography ref_ids to bibliography entry nodes
        :param ref_entry_nodes: mapping from reference ref_ids to reference entry nodes
        :return: the paragraph node
        """
        paragraph_node = Node(
            content=paragraph_json["text"],
            ntype=NTYPE_PARAGRAPH
        )

        intertext_document.add_node(paragraph_node)

        parent_edge = Edge(
            src_node=section_node,
            tgt_node=paragraph_node,
            etype=Etype.PARENT
        )
        next_edge = Edge(
            src_node=pred_node,
            tgt_node=paragraph_node,
            etype=Etype.NEXT
        )

        intertext_document.add_edge(parent_edge)
        intertext_document.add_edge(next_edge)

        for cite_span in paragraph_json["cite_spans"]:
            self._parse_cite_span(cite_span, intertext_document, paragraph_node, bib_entry_nodes)

        for ref_span in paragraph_json["ref_spans"]:
            self._parse_ref_span(ref_span, intertext_document, paragraph_node, ref_entry_nodes)

        return paragraph_node

    def _parse_cite_span(self, cite_span_json: Dict[str, Any],
                         intertext_document: IntertextDocument, paragraph_node: Node,
                         bib_entry_nodes: Dict[str, Node]) -> None:
        """
        Parse the given S2ORC cite span.

        {"start": int, "end": int, "text": str, "ref_id": str}

        :param cite_span_json: S2ORC cite span
        :param intertext_document: intertext document object
        :param paragraph_node: paragraph that the cite span is part of
        :param bib_entry_nodes: mapping from bibliography ref_ids to bibliography entry nodes
        """
        cite_span_node = SpanNode(
            ntype=NTYPE_CITE_SPAN,
            src_node=paragraph_node,
            start=cite_span_json["start"],
            end=cite_span_json["end"] - 1  # end is inclusive with ITG but exclusive with S2ORC
        )
        intertext_document.add_node(cite_span_node)

        # link edge between paragraph node and cite span node is created automatically

        edge_to_bib_entry = Edge(
            src_node=cite_span_node,
            tgt_node=bib_entry_nodes[cite_span_json["ref_id"]],
            etype=Etype.LINK
        )
        intertext_document.add_edge(edge_to_bib_entry)

    def _parse_ref_span(self, ref_span_json: Dict[str, Any],
                        intertext_document: IntertextDocument, paragraph_node: Node,
                        ref_entry_nodes: Dict[str, Node]):
        """
        Parse the given S2ORC reference span.

        {"start": int, "end": int, "text": str, "ref_id": str}

        :param ref_span_json: S2ORC reference span
        :param intertext_document: intertext document object
        :param paragraph_node: paragraph that the reference span is part of
        :param ref_entry_nodes: mapping from reference ref_ids to reference entry nodes
        """
        ref_span_node = SpanNode(
            ntype=NTYPE_REF_SPAN,
            src_node=paragraph_node,
            start=ref_span_json["start"],
            end=ref_span_json["end"] - 1  # end is inclusive with ITG but exclusive with S2ORC
        )
        intertext_document.add_node(ref_span_node)

        # link edge between paragraph node and reference span node is created automatically

        edge_to_ref_entry = Edge(
            src_node=ref_span_node,
            tgt_node=ref_entry_nodes[ref_span_json["ref_id"]],
            etype=Etype.LINK
        )
        intertext_document.add_edge(edge_to_ref_entry)

    def _parse_bib_entry(self, bib_entry_id: str, bib_entry_json: Dict[str, Any],
                         intertext_document: IntertextDocument, bib_entries_section_node: Node, pred_node,
                         bib_entry_nodes: Dict[str, Node]) -> Node:
        """
        Parse the given S2ORC bibliography entry.

        {
            "title": str,
            "authors": [{"first": str, "middle": [str], "last": str, "suffix": str}],
            "year": str,
            "venue": str,
            "link": str
        }

        :param bib_entry_id: ref_id of the given S2ORC bibliography entry
        :param bib_entry_json: S2ORC bibliography entry
        :param intertext_document: intertext document object
        :param bib_entries_section_node: bibliography entries' section node
        :param pred_node: bibliography entry's predecessor node
        :param bib_entry_nodes: mapping from bibliography ref_ids to bibliography entry nodes
        :return: the bibliography entry node
        """
        bib_entry_node = Node(
            content=bib_entry_json["title"],  # TODO: maybe put fully-formatted bibliography entry as content
            ntype=NTYPE_BIB_ENTRY,
            meta={
                "title": bib_entry_json["title"],
                "authors": bib_entry_json["authors"],
                "year": bib_entry_json["year"],
                "venue": bib_entry_json["venue"],
                "link": bib_entry_json["link"],
                "ref_id": bib_entry_id
            }
        )
        intertext_document.add_node(bib_entry_node)

        bib_entry_parent_edge = Edge(
            src_node=bib_entries_section_node,
            tgt_node=bib_entry_node,
            etype=Etype.PARENT
        )
        intertext_document.add_edge(bib_entry_parent_edge)

        bib_entry_next_edge = Edge(
            src_node=pred_node,
            tgt_node=bib_entry_node,
            etype=Etype.NEXT
        )
        intertext_document.add_edge(bib_entry_next_edge)

        bib_entry_nodes[bib_entry_id] = bib_entry_node
        return bib_entry_node

    def _parse_ref_entry(self, ref_entry_id: str, ref_entry_json: Dict[str, Any],
                         intertext_document: IntertextDocument, ref_entries_section_node: Node, pred_node,
                         ref_entry_nodes: Dict[str, Node]) -> Node:
        """
        Parse the given S2ORC reference entry.

        {
            "text": str,
            "type": str
        }

        :param ref_entry_id: ref_id of the given S2ORC reference entry
        :param ref_entry_json: S2ORC reference entry
        :param intertext_document: intertext document object
        :param ref_entries_section_node: reference entries' section node
        :param pred_node: reference entry's predecessor node
        :param ref_entry_nodes: mapping from reference ref_ids to reference entry nodes
        :return: the reference entry node
        """
        ref_entry_node = Node(
            content=ref_entry_json["text"],
            ntype=NTYPE_REF_ENTRY,
            meta={
                "type": ref_entry_json["type"],
                "ref_id": ref_entry_id
            }
        )
        intertext_document.add_node(ref_entry_node)

        ref_entry_parent_edge = Edge(
            src_node=ref_entries_section_node,
            tgt_node=ref_entry_node,
            etype=Etype.PARENT
        )
        intertext_document.add_edge(ref_entry_parent_edge)

        ref_entry_next_edge = Edge(
            src_node=pred_node,
            tgt_node=ref_entry_node,
            etype=Etype.NEXT
        )
        intertext_document.add_edge(ref_entry_next_edge)

        ref_entry_nodes[ref_entry_id] = ref_entry_node
        return ref_entry_node


def process_file_pair(input_tuple):
    """
    Transform a random sample of valid documents in the given S2ORC shard into ITG format.

    This script takes as input the file paths of one shard of the S2ORC corpus and writes its output to one JSONL file
    of the S2ORC-ITG-Subset.

    :param input_tuple: (metadata_file_path, pdf_parse_file_path, paper_ids_path, output_path, num_documents_per_shard)
    :return: number of transformation failures
    """
    metadata_fp, pdf_parse_fp, paper_ids_path, output_path, num_documents_per_shard = input_tuple

    # load the paper ids
    with open(paper_ids_path, "r", encoding="utf-8") as file:
        paper_ids = json.load(file)

    shard_id = metadata_fp[metadata_fp.rindex("_") + 1:metadata_fp.rindex(".jsonl")]
    paper_ids = paper_ids[shard_id]
    paper_ids = paper_ids[:num_documents_per_shard + 100]  # 100 buffer in case of failures
    paper_ids_set = set(paper_ids)

    # load metadata but keep only the required metadata
    metadata_jsonl = []
    with gzip.open(metadata_fp, "rb") as file:
        reader = io.BufferedReader(file)
        for line in reader.readlines():
            metadata_json = json.loads(line)
            if metadata_json["paper_id"] in paper_ids_set:
                metadata_jsonl.append(metadata_json)

    # load PDF parses but keep only the required PDF parses
    pdf_parse_jsonl = []
    with gzip.open(pdf_parse_fp, "rb") as file:
        reader = io.BufferedReader(file)
        for line in reader.readlines():
            pdf_parse_json = json.loads(line)
            if pdf_parse_json["paper_id"] in paper_ids_set:
                pdf_parse_jsonl.append(pdf_parse_json)

    # convert the documents to ITG format
    intertext_documents = []
    num_failures = 0
    for paper_id in paper_ids:
        try:
            parser = S2ORCParser(
                metadata_path=metadata_fp,
                pdf_parse_path=pdf_parse_fp,
                paper_id=paper_id
            )

            intertext_document = parser(
                metadata_jsonl=metadata_jsonl,
                pdf_parse_jsonl=pdf_parse_jsonl
            )
            intertext_documents.append(intertext_document)
        except Exception as e:
            num_failures += 1
            logger.warning(repr(e))

        if len(intertext_documents) == num_documents_per_shard:
            break

    # save the intertext documents
    path = os.path.join(output_path, f"{shard_id}.jsonl")
    with open(path, "w", encoding="utf-8") as file:
        for ix, intertext_document in enumerate(intertext_documents):
            json_str = intertext_document.to_json(indent=None)
            file.write(json_str)
            if ix != len(intertext_documents) - 1:
                file.write("\n")

    return num_failures
