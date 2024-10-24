import { AfterViewInit, Component, ViewChild } from '@angular/core';
import { MessageService } from 'primeng/api';
import { FileUploadEvent } from 'primeng/fileupload';
import { QaService } from './qa.service';
import { Document } from './dto/document';
import { qa } from './dto/qa';
import { Table } from 'primeng/table';

interface UploadEvent {
  originalEvent: Event;
  files: File[];
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements AfterViewInit {
  public questionCount: number = 5;
  public startPage: number = 1;
  public endPage: number = 10;
  public maxEndPage: number = 99999;
  private docs: Document[] = [];
  public qas: qa[] = []

  public isUploading: boolean = false
  public uploaded: boolean = false
  public isGenerating: boolean = false

  @ViewChild('table') table: Table;
  cols = [
    { field: 'question', header: 'Question' },
    { field: 'answer', header: 'Answer' },
  ];
constructor(private messageService: MessageService, private qaService: QaService) { }
  ngAfterViewInit(): void {
    // for (let i = 0 ; i < 100 ; i++)
    //   this.qas.push(new qa("Test", "Test"))
  }

onBeforeUpload() {
  console.log('before upload');
  this.isUploading = true;
}

onUpload(event: FileUploadEvent) {
  if (!event.files || event.files.length == 0)
    return;
  let file: File = null;
  file = event.files[0];


  this.qaService.anlyze_file_content(file).subscribe(docs => {
    console.log(docs);
    this.docs = docs
    this.maxEndPage = docs.length;
    this.endPage = this.maxEndPage;
    if (this.startPage > this.maxEndPage)
      this.startPage = this.maxEndPage;
    this.isUploading = false;
    this.uploaded = true;
    this.messageService.add({ severity: 'info', summary: 'Success', detail: 'File Uploaded with Basic Mode' });
  });

}

generateQA() {
  const generate_docs = this.docs.slice(this.startPage - 1, this.endPage);
  this.isGenerating = true;
  this.qaService.generate_qa(this.questionCount, generate_docs).subscribe(qas => {
    for (let nqa of qas) {
      if (!this.qas.some(p => p.equals(nqa))) {
        this.qas.push(new qa(nqa.question, nqa.answer));
      }
    }
    this.isGenerating = false;
    this.messageService.add({ severity: 'info', summary: 'Success', detail: 'Generate QA Success' });
  });
}


  export () {
  this.table.exportCSV()
}
}