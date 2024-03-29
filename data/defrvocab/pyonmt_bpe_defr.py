import pyonmttok
import gzip
import shutil
import os

tokenizer = pyonmttok.Tokenizer("aggressive", \
                                joiner_annotate = True, \
                                no_substitution = True, \
                                segment_alphabet_change = True, \
                                segment_alphabet = ['Han', 'Kanbun', 'Katakana', 'Hiragana'], \
                                segment_numbers = True, \
                                preserve_segmented_tokens = True, \
                                support_prior_joiners = True)

learner = pyonmttok.BPELearner(tokenizer=tokenizer, symbols=32000, min_frequency=2)


for d in [
'de_fr/train_restricted/btxt_2dir_de-CH-fr-CH_Legal__Crawl2022-parlament-ch.de.gz',
'de_fr/train_restricted/btxt_2dir_de-CH-fr-CH_Misc__Crawl2022.de.gz',
'de_fr/train_restricted/btxt_2dir_de-CH-fr-CH_News__SwissAdmin-press-releases-20140611.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Dialog__OPUS-OpenSubtitles_0.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Dialog__OPUS-OpenSubtitles_1.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Dialog__OPUS-OpenSubtitles_2.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Dialog__tatoeba.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Dialog__TAUS-Idioms.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Dialog__ted-talks-fbk-r2.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Environment__EEA-GEMET-environment-definitions.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Finance__Crawl2022.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Finance__OPUS-ECB.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Finance__TildeModel-EESC.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__msdn-ui-strings.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__msdn-vs2005.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__OPUS-KDE4.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__TAUS-all-IT_0.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__TAUS-all-IT_1.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__DGT-TM.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2008.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2009.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2010.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2011.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2012.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2013.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2014.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2015.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2016.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2017.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2018.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2019.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2020.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__JRC-Acquis.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Manual__TAUS-IndMan-StrDoc.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Medical__ECDC-TM.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Medical__EMEA-Annexes.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Medical__MSD-Merck-Manual.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Medical__OPUS-EMEA.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Misc__Crawl2022.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__OPUS-WikiMedia.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_0.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_1.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_2.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_3.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_4.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_5.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_6.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_7.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_8.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Misc__WikiMatrix.de.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_News__Casmacat-GlobalVoices.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_News__france-blog-info.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_News__news-commentary.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Patent__patents-abstracts.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Patent__patents-claims.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Patent__patents-titles.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Proceedings__DCEP.de.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Proceedings__europarl.de.gz',
'de_fr/train_synthetic/gene_2dir_de-CH-fr-CH_Generic__currency-ISO.de.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-dates-daymonth.de.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-dates-prep.de.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-dates-short.de.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-version-nodate.de.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-versionnumber.de.gz',
'de_fr/train_synthetic/gene_2dir_de-XX-fr-YY_Misc__Formality-veuillez-merci-de.de.gz',
'de_fr/train_synthetic/gene_defr_de-XX-fr-YY_Generic__PN9-lingsynlist-money-currency-symbol.de.gz',
'de_fr/train_synthetic/gene_frde_de-XX-fr-YY_Generic__money-currency-symbol.de.gz',
'de_fr/train_synthetic/gene_frde_de-XX-fr-YY_Misc__Formality-coordination.de.gz',
'de_fr/train_synthetic/gene_frde_de-XX-fr-YY_Misc__Formality-subordination.de.gz',
'de_fr/train_synthetic/term_2dir_de-DE-fr-FR_Generic__20200505-PN9-lingsynlist-number-general.de.gz',
'de_fr/train/term_2dir_de-XX-fr-YY_Environment__EEA-GEMET-environment-labels.de.gz',
'de_fr/train_restricted/term_2dir_de-XX-fr-YY_Generic__enwiki-freebase-all-titles.de.gz',
'de_fr/train_restricted/term_2dir_de-XX-fr-YY_Generic__enwiki-namespaces.de.gz',
'de_fr/train/term_2dir_de-XX-fr-YY_Generic__IATE-EU-Terms.de.gz',
'de_fr/train_synthetic/term_2dir_de-XX-fr-YY_Generic__number-digits.de.gz',
'de_fr/train_synthetic/term_2dir_de-XX-fr-YY_Generic__number-ordinal.de.gz',
'de_fr/train_restricted/term_2dir_de-XX-fr-YY_IT__microsoft-localization.de.gz',
'de_fr/train_synthetic/term_2dir_de-XX-fr-YY_Names__20220930-PN9-person-names.de.gz',
'de_fr/train_restricted/term_defr_de-XX-fr-YY_Generic__enwiki-freebase-all-titles.de.gz',
'de_fr/train_restricted/term_defr_de-XX-fr-YY_Generic__enwiki-namespaces.de.gz',
'de_fr/train_restricted/term_defr_de-XX-fr-YY_Generic__Lookup-finaldict.de.gz',
'de_fr/train_synthetic/term_defr_de-XX-fr-YY_Misc__abbreviations.de.gz',
'de_fr/train_synthetic/term_frde_de-DE-fr-FR_Generic__20200505-PN9-lingsynlist-number-paragraph.de.gz',
'de_fr/train_restricted/term_frde_de-XX-fr-YY_Generic__enwiki-freebase-all-titles.de.gz',
'de_fr/train_restricted/term_frde_de-XX-fr-YY_Generic__enwiki-namespaces.de.gz',
'de_fr/train_restricted/term_frde_de-XX-fr-YY_Generic__Lookup-finaldict.de.gz',
'de_fr/train_synthetic/term_frde_de-XX-fr-YY_Misc__abbreviations.de.gz',
'de_fr/train_restricted/btxt_2dir_de-CH-fr-CH_Legal__Crawl2022-parlament-ch.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-CH-fr-CH_Misc__Crawl2022.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-CH-fr-CH_News__SwissAdmin-press-releases-20140611.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Dialog__OPUS-OpenSubtitles_0.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Dialog__OPUS-OpenSubtitles_1.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Dialog__OPUS-OpenSubtitles_2.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Dialog__tatoeba.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Dialog__TAUS-Idioms.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Dialog__ted-talks-fbk-r2.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Environment__EEA-GEMET-environment-definitions.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Finance__Crawl2022.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Finance__OPUS-ECB.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Finance__TildeModel-EESC.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__msdn-ui-strings.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__msdn-vs2005.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__OPUS-KDE4.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__TAUS-all-IT_0.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_IT__TAUS-all-IT_1.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__DGT-TM.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2008.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2009.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2010.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2011.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2012.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2013.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2014.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2015.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2016.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2017.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2018.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2019.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__EUR-Lex-2020.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Legal__JRC-Acquis.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Manual__TAUS-IndMan-StrDoc.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Medical__ECDC-TM.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Medical__EMEA-Annexes.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Medical__MSD-Merck-Manual.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Medical__OPUS-EMEA.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Misc__Crawl2022.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__OPUS-WikiMedia.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_0.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_1.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_2.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_3.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_4.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_5.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_6.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_7.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_Misc__paracrawl_8.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Misc__WikiMatrix.fr.gz',
'de_fr/train/btxt_2dir_de-XX-fr-YY_News__Casmacat-GlobalVoices.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_News__france-blog-info.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_News__news-commentary.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Patent__patents-abstracts.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Patent__patents-claims.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Patent__patents-titles.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Proceedings__DCEP.fr.gz',
'de_fr/train_restricted/btxt_2dir_de-XX-fr-YY_Proceedings__europarl.fr.gz',
'de_fr/train_synthetic/gene_2dir_de-CH-fr-CH_Generic__currency-ISO.fr.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-dates-daymonth.fr.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-dates-prep.fr.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-dates-short.fr.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-version-nodate.fr.gz',
'de_fr/train_synthetic/gene_2dir_de-DE-fr-FR_Generic__20210115-PN9-lingsynlist-versionnumber.fr.gz',
'de_fr/train_synthetic/gene_2dir_de-XX-fr-YY_Misc__Formality-veuillez-merci-de.fr.gz',
'de_fr/train_synthetic/gene_defr_de-XX-fr-YY_Generic__PN9-lingsynlist-money-currency-symbol.fr.gz',
'de_fr/train_synthetic/gene_frde_de-XX-fr-YY_Generic__money-currency-symbol.fr.gz',
'de_fr/train_synthetic/gene_frde_de-XX-fr-YY_Misc__Formality-coordination.fr.gz',
'de_fr/train_synthetic/gene_frde_de-XX-fr-YY_Misc__Formality-subordination.fr.gz',
'de_fr/train_synthetic/term_2dir_de-DE-fr-FR_Generic__20200505-PN9-lingsynlist-number-general.fr.gz',
'de_fr/train/term_2dir_de-XX-fr-YY_Environment__EEA-GEMET-environment-labels.fr.gz',
'de_fr/train_restricted/term_2dir_de-XX-fr-YY_Generic__enwiki-freebase-all-titles.fr.gz',
'de_fr/train_restricted/term_2dir_de-XX-fr-YY_Generic__enwiki-namespaces.fr.gz',
'de_fr/train/term_2dir_de-XX-fr-YY_Generic__IATE-EU-Terms.fr.gz',
'de_fr/train_synthetic/term_2dir_de-XX-fr-YY_Generic__number-digits.fr.gz',
'de_fr/train_synthetic/term_2dir_de-XX-fr-YY_Generic__number-ordinal.fr.gz',
'de_fr/train_restricted/term_2dir_de-XX-fr-YY_IT__microsoft-localization.fr.gz',
'de_fr/train_synthetic/term_2dir_de-XX-fr-YY_Names__20220930-PN9-person-names.fr.gz',
'de_fr/train_restricted/term_defr_de-XX-fr-YY_Generic__enwiki-freebase-all-titles.fr.gz',
'de_fr/train_restricted/term_defr_de-XX-fr-YY_Generic__enwiki-namespaces.fr.gz',
'de_fr/train_restricted/term_defr_de-XX-fr-YY_Generic__Lookup-finaldict.fr.gz',
'de_fr/train_synthetic/term_defr_de-XX-fr-YY_Misc__abbreviations.fr.gz',
'de_fr/train_synthetic/term_frde_de-DE-fr-FR_Generic__20200505-PN9-lingsynlist-number-paragraph.fr.gz',
'de_fr/train_restricted/term_frde_de-XX-fr-YY_Generic__enwiki-freebase-all-titles.fr.gz',
'de_fr/train_restricted/term_frde_de-XX-fr-YY_Generic__enwiki-namespaces.fr.gz',
'de_fr/train_restricted/term_frde_de-XX-fr-YY_Generic__Lookup-finaldict.fr.gz',
'de_fr/train_synthetic/term_frde_de-XX-fr-YY_Misc__abbreviations.fr.gz',
]:
  print(f'Processing {d}')
  if d.endswith('.gz'):
    print(f'Gunzipping {d} into {d}.txt')
    with gzip.open(d, 'rb') as inputF, open(f'{d}.txt', 'wb') as outputF:
      shutil.copyfileobj(inputF, outputF)
    print('Ingesting file...')
    learner.ingest_file(f'{d}.txt')
    print(f'Deleting {d}.txt')
    os.remove(f'{d}.txt')
  else:
    print('Ingesting file...')
    learner.ingest_file(d)

print('Creating BPE model "pyonmt_bpe_defr_32k_min2"')
tokenizer = learner.learn('pyonmt_bpe_defr_32k_min2')
