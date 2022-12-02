OUTPUT=../../../data/indexes/bm25
INPUT=../../../data/bm25_collection

if [ ! -f "$OUTPUT" ]; then
    echo "Creating index..."
    python -m pyserini.index -collection JsonCollection \
                            -generator DefaultLuceneDocumentGenerator \
                            -threads 20 \
                            -input ${INPUT} \
                            -index ${OUTPUT} \
							-storePositions -storeDocvectors -storeRaw
fi
