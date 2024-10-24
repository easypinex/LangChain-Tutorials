export class qa {
    answer: string;
    question: string;

    constructor(question: string, answer: string) {
        this.question = question
        this.answer = answer
    }

  equals(other: qa): boolean {
    return this.question === other.question;
  }
}
