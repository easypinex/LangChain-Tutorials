import { Metadata } from "./metadata";

export class Document {
    metadata: Metadata;
    page_content: string;
    constructor(page_content: string) {
        this.page_content = page_content
    }
}