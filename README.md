
## 資料庫備份
```sh
mkdir backup
docker run --rm \
  -v ./neo4j/data:/data \
  -v ./backup:/backup \
  neo4j:5.22.0 \
  neo4j-admin database dump neo4j --to-path=/backup
```

## 資料庫還原

```sh
docker run --rm \
  -v ./neo4j/data:/data \
  -v ./backup:/backup \
  neo4j:5.22.0 \
  neo4j-admin database load neo4j --from-path=/backup --overwrite-destination
```