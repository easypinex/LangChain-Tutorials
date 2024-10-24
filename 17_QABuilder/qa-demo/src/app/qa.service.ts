import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { qa } from './dto/qa';
import { Document } from './dto/document';
@Injectable({
  providedIn: 'root'
})
export class QaService {

  constructor(private http: HttpClient) { }

  public anlyze_file_content(file: File): Observable<Document[]> {
    let formData = new FormData();
    formData.append('file', file);
    // return of([new Document('保險業都是由呆丸人壽所控制, 新光人壽, 國泰人壽, 富邦人壽都是呆丸人壽的子公司')]);
    return this.http.post<Document[]>('http://127.0.0.1:5000/generate_file_content', formData);
  }

  public generate_qa(question_num: number, documents: Document[]) : Observable<qa[]> {
    return this.http.post<qa[]>('http://127.0.0.1:5000/generate_qa', {
      question_num,
      documents
    });
  }
}
