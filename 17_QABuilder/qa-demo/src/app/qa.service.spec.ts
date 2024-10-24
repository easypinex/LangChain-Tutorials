import { TestBed } from '@angular/core/testing';

import { QaService } from './qa.service';

describe('QaService', () => {
  let service: QaService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(QaService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
