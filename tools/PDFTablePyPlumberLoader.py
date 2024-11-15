from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document
import pandas as pd
import pdfplumber
from pdfplumber.page import CroppedPage, Page
from pdfplumber.table import Table
import logging
import os

logging = logging.getLogger("langchain")

class PDFTablePyPlumberLoader:
    def __init__(self, file_path, llm=None) -> None:
        self.file_path = file_path
        self.llm = llm

    def load(self, **open_kwargs) -> List[Document]:

        merge_table_candidates: List[Table] = []
        page_tables_list:List[List[Table]] = []
        
        document_packers:List[DocumentPacker] = []
        with pdfplumber.open(self.file_path, **open_kwargs) as pdf:
            pages = pdf.pages
            page_document_dict:Dict[Page, DocumentPacker] = {}
            for idx, page in enumerate(pages):
                tables = page.find_tables(table_settings={})
                croped_text_page = page
                for table in tables:
                    croped_text_page = croped_text_page.outside_bbox(table.bbox)
                crop_bbox = self._get_text_crop_box(page, tables)
                croped_text_page = croped_text_page.crop(crop_bbox)
                text = croped_text_page.extract_text()
                filename = os.path.basename(self.file_path)
                document = Document(text, metadata={"source": filename, "page_number": page.page_number})
                page_tables_list.append(tables)
                document_packer = DocumentPacker(document, self.llm)
                document_packers.append(document_packer)
                page_document_dict[page] =document_packer
                
                merge_table_candidates += self._get_merge_candidate(croped_text_page, tables)                
                
                for table in tables:
                    # text = table.extract() # for debug
                    if table in merge_table_candidates:
                        continue
                    # 代表表格不在底部
                    if len(merge_table_candidates) > 0 and table.bbox[1] - page.height / 15 < 0:
                        # 有待合併清單, 且當前表格接近上方
                        first_table = merge_table_candidates[0]
                        for merge_table_candi in merge_table_candidates:
                            page_document_dict[first_table.page].tables.append(merge_table_candi)
                        merge_table_candidates.clear()
                    else:
                        document_packer.tables.append(table)
                        
        if len(merge_table_candidates) > 0:
            # 還有剩餘的待合併清單
            first_table = merge_table_candidates[0]
            for merge_table_candi in merge_table_candidates:
                page_document_dict[first_table.page].tables.append(merge_table_candi)
            merge_table_candidates.clear()
            
        result: List[Document] = []
        llm_tasks: List[Tuple[DocumentPacker, str]] = []
        for documnet_packer in document_packers:
            result.append(documnet_packer.table_parse())
            if self.llm:
                llm_tasks += documnet_packer.llm_tasks
        if self.llm:
            self._documnet_table_desc_with_llm(llm_tasks)
        return result
    
    def _get_merge_candidate(self, page: CroppedPage, tables: List[Table]) -> List[Table]:
        '''取得可能需要合併的表格候選人 通常代表表格接近底部'''
        candidates = []
        for table in tables:
            if (table.bbox[3]+ page.height / 15) > page.height:
                # 如果表格的下邊界接近底部
                candidates.append(table)
        return candidates

    def _documnet_table_desc_with_llm(self, llm_tasks: List[Tuple['DocumentPacker', str]]):
        '''將表格的描述加入到該頁結尾'''
        from pydantic import BaseModel, Field
        from langchain_core.prompts import ChatPromptTemplate
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        class KnowledgeEntity(BaseModel):
            description: str = Field(
                description="Markdown格式描述的完整資訊知識"
            )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一個資訊整理專家，以Markdown格式輸出以下知識，但是不使用表格輸出，並包含完整資訊，以繁體中文回應",
            ),
            (
                "human",
                """請整理以下表格內容，並以文字輸出，並保留完整知識\n{list_str}""",
            ),
        ])
        desc_llm = self.llm.with_structured_output(
            KnowledgeEntity
        )
        desc_chain = prompt | desc_llm
        
        def table_description(document_packer: DocumentPacker, list_str: str):
            try:
                desc = desc_chain.invoke({"list_str": list_str}).description
                if desc:
                    document_packer.document.page_content += '\n\n' + desc
            except Exception as e:
                print(e)
                document_packer.document.page_content += '\n\n' + list_str
                

        futures = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submitting all tasks and creating a list of future objects
            for document_packer, list_str in llm_tasks:
                future = executor.submit(table_description, document_packer, list_str)
                futures.append(future)
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing documents"
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f'process element faild!, error:\n{e}')
    
    def _get_text_crop_box(self, page, tables):
        '''
        將邊界及頁首頁尾去除, 取得剩餘的 bbox
        
        需考慮Table邊界, 若Table邊界在非常邊邊, 則不切Table為主
        '''
        crop_bbox = (page.width/20, page.height/20, page.width-page.width/20, page.height - page.height/20) # (x0, top, x1, bottom)
        # 避免 crop 到表格, 最大crop到表格寬高
        table_x0_min = 999999
        table_x1_max = 0
        table_y0_min = 999999
        table_y0_max = 0
        for table in tables:
            table_x0_min = min(table_x0_min, table.bbox[0])
            table_y0_min = min(table_y0_min, table.bbox[1])
            table_x1_max = max(table_x1_max, table.bbox[2])
            table_y0_min = max(table_x1_max, table.bbox[3])
        crop_bbox = (min(crop_bbox[0], table_x0_min), min(crop_bbox[1], table_y0_min), max(crop_bbox[2], table_x1_max), max(crop_bbox[3], table_y0_max))
        return crop_bbox
        
    def _get_merge_top_talbe(self, total_height, merge_table_candidate: List[Table]) -> Table | None:
        '''
        取得貼近頁面頂端的表格
        '''
        for idx, table in enumerate(merge_table_candidate):
            if table.bbox[1] - total_height / 15 < 0:
                return merge_table_candidate.pop(idx)
    
    
class DocumentPacker:
    def __init__(self, document, llm):
        self.document: Document = document
        self.llm = llm
        self.tables: List[Table] = []
        self.tableareas: List[TableArea] = []
        self.llm_tasks: List[Tuple['DocumentPacker', str]] = []
        
    def table_parse(self) -> Document:
        self.calculate_sub_tables()
        for table in self.tableareas:
            df = pd.DataFrame(table.table.extract())
            concat_str = str(df.values.tolist())
            if self.llm:
                self.llm_tasks.append((self, concat_str))
            else:
                self.document.page_content += '\n\n' + concat_str
        return self.document
        
    def calculate_sub_tables(self):
        '''
        為了解決表格中還有表格的問題, 將小的Table歸屬於大Table
        
        因為暫時不處理表格中的表格(大表格還是會有小表格的文字, 只是失去小表格的內容), 
        
        這裏只是歸類
        '''
        tableareas: List[TableArea] = [TableArea(table) for table in self.tables]
        tableareas_sorted = sorted(tableareas, key=lambda table: table.area)
        # 遍歷所有的 tableareas，檢查每個表格是否位於另一個更大的表格中
        for i, table in enumerate(tableareas_sorted):
            for larger_table in tableareas_sorted[i+1:]:
                # 如果當前的表格嵌套在 larger_table 中，則將其歸屬於 larger_table 的子表格（sub_tables）
                if table.in_other_table(larger_table):
                    larger_table.sub_tables.append(table)
                    table.parent_table = larger_table
                    self.tables.remove(table.table)
                    break  # 一旦表格被歸屬，停止當前表格的進一步檢查
        self.tableareas = [table for table in tableareas]

class TableArea:
    def __init__(self, table: Table):
        self.table = table
        self.sub_tables: List[TableArea] = []
        self.parent_table: TableArea = None
    
    def in_other_table(self, other: 'TableArea'):
        if self.area >= other.area:
            return False
        if self.x1 >= other.x1 and self.x2 <= other.x2 and self.y1 >= other.y1 and self.y2 <= other.y2:
            return True
        return False
    
    @property
    def x1(self):
        return self.table.bbox[0]
    
    @property
    def y1(self):
        return self.table.bbox[1]
    
    @property
    def x2(self):
        return self.table.bbox[2]
    
    @property
    def y2(self):
        return self.table.bbox[3]
    
    @property
    def width(self):
        return self.x2 - self.x1
    
    @property
    def height(self):
        return self.y2 - self.y1
    
    @property
    def area(self):
        return self.width * self.height


if __name__ == '__main__':
    import os
    file_path = os.path.join('..', 'data', '新契約個人保險投保規則手冊-核保及行政篇(113年7月版)_業務通路版.pdf')
    loader = PDFTablePyPlumberLoader(file_path)
    pages = loader.load(pages=[8, 9, 10, 11, 12])
    for page in pages:
        print(page.page_content)
        print('-' * 40)