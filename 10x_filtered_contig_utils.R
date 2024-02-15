library(dplyr)

aa_cdist <- function(contig_dat, consensus_dat){
  contig_dat <- contig_dat %>% add_cdr_concat_cols()

  comb_cons_sub <- consensus_dat %>% 
          add_cdr_concat_cols() %>%
          rename(raw_consensus_id = consensus_id, consensus_cdr_fwr = cdr_fwr, consensus_cdr_concat = cdr_concat, consensus_cdr3 = cdr3) %>%
          mutate(raw_consensus_id = gsub("consensus", "consensus_", raw_consensus_id)) %>%
          select(raw_consensus_id, consensus_cdr_fwr, consensus_cdr_concat, consensus_cdr3)
  
  contig_dat <- contig_dat %>%
          left_join(comb_cons_sub)
  
  #aa_cdist from this ref - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008391
  contig_dat <- contig_dat %>%
          mutate(aa_cdist = sapply(1:nrow(.),function(i){
    adist(cdr_fwr[i], consensus_cdr_fwr[i])[1,1]
  }))

}

convert_to_paired <- function(contig_dat){
   cols_dont_add_prefix <- c("barcode", "is_cell", "raw_clonotype_id")
   contig_dat_hc <- contig_dat %>% filter(chain == "IGH")
   contig_dat_lc <- contig_dat %>% filter(chain %in% c("IGK", "IGL"))

   colnames(contig_dat_hc)[!colnames(contig_dat_hc) %in% cols_dont_add_prefix] <-
           paste0("hc_", colnames(contig_dat_hc)[!colnames(contig_dat_hc) %in% cols_dont_add_prefix])
   colnames(contig_dat_lc)[!colnames(contig_dat_lc) %in% cols_dont_add_prefix] <- 
           paste0("lc_", colnames(contig_dat_lc)[!colnames(contig_dat_lc) %in% cols_dont_add_prefix])
   
   contig_dat_combined <- left_join(contig_dat_hc, contig_dat_lc)
}

add_cdr_concat_cols <- function(dat){
  dat <- dat %>%
        mutate(cdr_fwr = paste0(fwr1, cdr1, fwr2, cdr2, fwr3, cdr3, fwr4)) %>%
        mutate(cdr_concat = paste0(cdr1, cdr2, cdr3))

}
