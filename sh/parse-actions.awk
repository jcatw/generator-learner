BEGIN {
    FS = ":"
    OFS = ","
    inep = 0
    iter=1
}
($3 ~ /episode/ && $4 == ep) {inep = 1}
($3 ~ /episode/ && $4 != ep) {inep = 0}
(inep && $3 ~ /.*action/ && $3 !~ /.*dictionary/) {print iter,$4; iter++}

