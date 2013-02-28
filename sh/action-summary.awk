BEGIN {
    FS = ":"
    OFS = ","
    inep = 0
}
($3 ~ /episode/ && $4 == ep) {inep = 1}
($3 ~ /episode/ && $4 != ep) {inep = 0}
(inep && $3 ~ /.*action/ && $3 !~ /.*dictionary/) {acount[$4] += 1}
END {
    for (a in acount) {
	print a, acount[a]
	cumsum += acount[a]
    }
    print " total",cumsum
}
