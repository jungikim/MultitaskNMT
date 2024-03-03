
import numpy as np
import tensorflow as tf
from opennmt import constants, tokenizers

_sentinelTokens = ['｟NOT_SENTINEL｠']
_sentinelTokens.extend([ '｟SENTINEL＃'+str(idx)+'｠' for idx in range(1,3001) ])

class generateSpanCorruption():
    def __init__(self, noise_density=0.15,#15,
                       mean_noise_span_length=3.0,#3.0,
                       sentinelTokens = _sentinelTokens):

        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.tokenizer = tokenizers.SpaceTokenizer()
        self.sentinelTokens = np.array(sentinelTokens)


    def t5_single_example_denoise(self,
                            features,
                            *, # this forces the caller to use named arguments
                            noise_mask_fn,
                            inputs_fn,
                            targets_fn):
        element = features

        tokens = self.tokenizer.tokenize(element)
        if isinstance(tokens, tf.RaggedTensor):
            length = tokens.row_lengths()
            tokens = tokens.to_tensor(default_value=constants.PADDING_TOKEN)
        else:
            length = np.shape(tokens)[0]

        noise_mask = noise_mask_fn(length)

        inputs = inputs_fn(tokens, noise_mask)

        if targets_fn:
            targets = targets_fn(tokens, noise_mask)
        else:
            targets = tokens

        return {
            'src': inputs,
            'tgt': targets
        }


    def noise_span_to_unique_sentinel(self, tokens, noise_mask):#, seed):
        #del seed
        prev_token_is_noise     = np.pad(noise_mask[:-1], [[1, 0]])
        first_noise_tokens      = np.logical_and(noise_mask, np.logical_not(prev_token_is_noise))
        subsequent_noise_tokens = np.logical_and(noise_mask, prev_token_is_noise)

        # sentinel = self.sentinel_id() + 1 - np.cumsum(np.cast(first_noise_tokens, tokens.dtype))
        sentinel = np.cumsum(np.cast['int32'](first_noise_tokens))
        sentinel = self.sentinelTokens[sentinel]
        tokens = np.where(first_noise_tokens, sentinel, tokens)
        # return tf.boolean_mask(tokens, np.logical_not(subsequent_noise_tokens))
        return np.ma.masked_array(tokens, mask=subsequent_noise_tokens).compressed()


    def nonnoise_span_to_unique_sentinel(self, tokens, noise_mask):#, seeds):
        return self.noise_span_to_unique_sentinel(tokens, np.logical_not(noise_mask))#, seeds=seeds)


    def random_spans_noise_mask(self, length, random_roll=True):#, seeds):
        if self.noise_density == 0.0:
            return np.zeros(length, np.bool)

        orig_length = length
        # increase length to avoid degeneracy
        length = np.maximum(length, 2)

        def to_int(x):
            return np.cast['int32'](x)

        def to_float(x):
            return np.cast['float32'](x)

        num_noise_tokens = to_int(np.round(to_float(length) * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1), length - 1)

        num_noise_spans = to_int(np.round(to_float(num_noise_tokens) / self.mean_noise_span_length))
        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = np.maximum(num_noise_spans, 1)

        num_nonnoise_tokens = length - num_noise_tokens


        def seqio_stateless_shuffle_np(value):#, seed):
            flat_value = np.reshape(value, [-1])
            indices = np.argsort(np.random.uniform(size=np.shape(flat_value)))
            flat_shuffle = np.take_along_axis(flat_value, indices, axis=0)
            return np.reshape(flat_shuffle, np.shape(value))

        def segment_sum(data, segment_ids):
            data = np.asarray(data)
            s = np.zeros((np.max(segment_ids)+1,) + data.shape[1:], dtype=data.dtype)
            np.add.at(s, segment_ids, data)
            return s

        def unsorted_segment_sum(data, segment_ids, num_segments):
            data = np.asarray(data)
            s = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
            np.add.at(s, segment_ids, data)
            return s


        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):#, seed):
            first_in_segment = np.pad(
                          seqio_stateless_shuffle_np(
                              to_int(np.arange(num_items - 1) < num_segments - 1)
                              ),#,seed),
                          [[1, 0]])
            segment_id = np.cumsum(first_in_segment)

            segment_length = segment_sum(np.ones_like(segment_id), segment_id)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)#, seeds[0])
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)#, seeds[1])
        interleaved_span_lengths = np.reshape(
                                        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
                                        [num_noise_spans * 2])
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        # span_start_indicator = tf.math.unsorted_segment_sum(np.ones_like(span_starts), span_starts, length)
        span_start_indicator = unsorted_segment_sum(np.ones_like(span_starts), span_starts, length)
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        mask = is_noise[:orig_length]

        if random_roll:
            # roll_seed = (seeds[0][0]+seeds[1][1], seeds[0][1]-seeds[1][0])  # new seed.
            # Roll the mask by a random offset e.g. for offset=2: [1,2,3,4] => [3,4,1,2]
            offset = tf.random.uniform(
                [1], dtype=tf.int32, minval=0, maxval=length)[0] #seed=roll_seed, 
            mask = tf.roll(mask, shift=offset, axis=0)


        return mask



def run_main():
    generator = generateSpanCorruption()

    #features = 'Hello , World ! This is a test sentence . Hello , World ! This is a test sentence .'
    features = "Chapter I: The Creation of the World The First Things Created - The Alphabet - The First Day - The Second Day - The Third Day - The Fourth Day - The Fifth Day - The Sixth Day - All Things Praise the Lord Chapter II: Adam Man and the World - The Angels and the Creation of Man - The Creation of Adam - The Soul of Man - The Ideal Man - The Fall of Satan - Woman - Adam and Eve in Paradise - The Fall of Man - The Punishment - Sabbath in Heaven - Adam's Repentance - The Book of Raziel - The Sickness of Adam - Eve's Story of the Fall - The Death of Adam - The Death of Eve Chapter III: The Ten Generations The Birth of Cain - Fratricide - The Punishment of Cain - The Inhabitants of the Seven Earths - The Descendants of Cain - The Descendants of Adam and Lilith - Seth and His Descendants - Enosh - The Fall of the Angels - Enoch, Ruler and Teacher - The Ascension of Enoch - The Translation of Enoch - Methuselah Chapter IV: Noah The Birth of Noah - The Punishment of the Fallen Angels - The Generation of the Deluge - The Holy Book - The Inmates of the Ark - The Flood - Noah Leaves the Ark - The Curse of Drunkenness - Noah's Descendants Spread Abroad - The Depravity of Mankind - Nimrod - The Tower of Babel Chapter V: Abraham The Wicked Generations - The Birth of Abraham - The Babe Proclaims God - Abraham's First Appearance in Public - The Preacher of the True Faith - In the Fiery Furnace - Abraham Emigrates to Haran - The Star in the East - The True Believer - The Iconoclast - Abraham in Canaan - His Sojourn in Egypt - The First Pharaoh - The War of the Kings - The Covenant of the Pieces - The Birth of Ishmael - The Visit of the Angels - The Cities of Sin - Abraham Pleads for the Sinners - The Destruction of the Sinful Cities - Among the Philistines - The Birth of Isaac - Ishmael Cast Off - The Two Wives of Ishmael - The Covenant with Abimelech - Satan Accuses Abraham - The Journey to Moriah - The Akedah - The Death and Burial of Sarah - Eliezer's Mission - The Wooing of Rebekah - The Last Years of Abraham - A Herald of Death - Abraham Views Earth and Heaven - The Patron of Hebron Chapter VI: Jacob The Birth of Esau and Jacob - The Favorite of Abraham - The Sale of the Birthright - Isaac with the Philistines - Isaac Blesses Jacob - Esau's True Character Revealed - Jacob Leaves His Father's House - Jacob Pursued by Eliphaz and Esau - The Day of Miracles - Jacob with Laban - The Marriage of Jacob - The Birth of Jacob's Children - Jacob Flees before Laban - The Covenant with Laban - Jacob and Esau Prepare to Meet - Jacob Wrestles with the Angel - The Meeting between Esau and Jacob - The Outrage at Shechem - A War Frustrated - The War with the Ninevites - The War with the Amorites - Isaac Blesses Levi and Judah - Joy and Sorrow in the House of Jacob - Esau's Campaign against Jacob - The Descendants of Esau Chapter I: Joseph The Favorite Son - Joseph Hated by His Brethren - Joseph Cast into the Pit - The Sale - Joseph's Three Masters - Joseph's Coat Brought to His Father - Judah and His Sons - The Wives of the Sons of Jacob - Joseph the Slave of Potiphar - Joseph and Zuleika - Joseph Resists Temptation - Joseph in Prison - Pharaoh's Dreams - Joseph before Pharaoh - The Ruler of Egypt - Joseph's Brethren in Egypt - Joseph Meets His Brethren - The Second Journey to Egypt - Joseph and Benjamin - The Thief Caught - Judah Pleads and Threatens - Joseph Makes Himself Known - Jacob Receives the Glad Tidings - Jacob Arrives in Egypt - Joseph's Kindness and Generosity - Jacob's Last Wish - The Blessing of Ephraim and Manasseh - The Blessing of the Twelve Tribes - The Death of Jacob - The Sons of Jacob at War with the Sons of Esau - Zepho King of Kittim - The Nations at War - Joseph's Magnanimity - Asenath - The Marriage of Joseph - Kind and Unkind Brethren - Treachery Punished - The Death and Burial of Joseph Chapter II: The Sons of Jacob Significant Names - Reuben's Testament - Simon's Admonition against Envy - The Ascension of Levi - Judah Warns against Greed and Unchastity - Issachar's Singleness of Heart - Zebulon Exhorts unto Compassion - Dan's Confession - Naphtali's Dreams of the Division of the Tribes - Gad's Hatred - Asher's Last Words - Benjamin Extols Joseph Chapter III: Job Job and the Patriarchs - Job's Wealth and Benefactions - Satan and Job - Job's Suffering - The Four Friends - Job Restored Chapter IV: Moses in Egypt The Beginning of the Egyptian Bondage - Pharaoh's Cunning - The Pious Midwives - The Three Counsellors - The Slaughter of the Innocents - The Parents of Moses - The Birth of Moses - Moses Rescued from the Water - The Infancy of Moses - Moses Rescued by Gabriel - The Youth of Moses - The Flight - The King of Ethiopia - Jethro - Moses Marries Zipporah - A Bloody Remedy - The Faithful Shepherd - The Burning Thorn-bush - The Ascension of Moses - Moses Visits Paradise and Hell - Moses Declines the Mission - Moses Punished for His Stubbornness - The Return to Egypt - Moses and Aaron before Pharaoh - The Suffering Increases - Measure for Measure - The Plagues Brought through Aaron - The Plagues Brought through Moses - The First Passover - The Smiting of the First-born - The Redemption of Israel from Egyptian Bondage - The Exodus Chapter I The Long Route - Pharaoh Pursues the Hebrews - The Sea Divided - The Passage through the Red Sea - The Destruction of the Egyptians - The Song at the Sea - The Awful Desert - The Heavenly Food - The Gathering of the Manna - Miriam' s Well - Amalek's War against Israel - Amalek Defeated - Jethro Chapter II Installation of Elders - Jethro Rewarded - The Time is at Hand - The Gentiles Refuse the Torah - The Contest of the Mountains - The Torah Offered to Israel - Israel Prepares for the Revelation - The Revelation on Mount Sinai - The First Commandment - The Other Commandments Revealed on Sinai - The Unity of the Ten Commandments - Moses Chosen as Intermediator - Moses and the Angels Strive for the Torah - Moses Receives the Torah - The Golden Calf - Moses Blamed for Israel's Sin - The Punishment of the Sinners - Moses Intercedes for the People - The Inscrutable Ways of the Lord - The Thirteen Attributes of God - The Second Tables - The Census of the People - The Erection of the Tabernacle Commanded Chapter III The Materials for the Construction of the Tabernacle - Bezalel - The Ark with the Cherubim - The Table and the Candlestick - The Altar - The Symbolical Significance of the Tabernacle - The Priestly Robes - The Stones in the Breastplate - The Completion of the Tabernacle - The Setting up of the Tabernacle - The Consecration of the Priests - The Day of the Ten Crowns - The Interrupted Joy - The Gifts of the Princes - The Revelations in the Tabernacle - The Cleansing of the Camp - The Lighting of the Candlestick Chapter IV The Twelve Princes of the Tribes - The Census of the Levites - The Four Divisions of the Levites - The Four Standards - The Camp - The Blasphemer and the Sabbath-breaker - The Ungrateful Multitude - The Flesh-pots of Egypt - The Appointment of the Seventy Elders - Eldad and Medad - The Quails - Aaron and Miriam Slander Moses - Miriam's Punishment - The Sending of the Spies - Significant Names - The Spies in Palestine - The Slanderous Report - The Night of Tears - Ingratitude Punished - The Years of Disfavor Chapter V The Rebellion of Korah - Korah Abuses Moses and the Torah - Moses Pleads in Vain with Korah - Korah and His Horde Punished - On and the Three Sons of Korah Saved - Israel Convinced of Aaron's Priesthood - The Waters of Meribah - Moses' Anger Causes His Doom - Edom's Unbrotherly Attitude toward Israel - The Three Shepherds - Preparing Aaron for Impending Death - Aaron's Death - The General Mourning for Aaron - The False Friends - The Brazen Serpent - At Arnon - Sihon, the King of the Amorites - The Giant Og - Moses' Speech of Admonition - Balak, King of Moab Chapter VI Balaam, the Heathen Prophet - Balak's Messengers to Balaam - Balaam Accepts Balak's Invitation - Baiaam's Ass - Balaam Runs into His Own Destruction - Balaam with Balak - Balaam's Sacrifices Refused - Balaam Extols Israel - Balaam's Hopes Disappointed - Curses Turned into Blessings - Balaam's Wicked Counsel - Phinehas, Zealous for God - Twelve Miracles - Phinehas Rewarded - The Daughters of Zelophmehad - The appointment of Joshua - Moses' Legacy to Joshua - Moses' last campaign - The Complete Annihilation of Midian - The Gruesome End of Balaam - The Victorious Return from the War - Wealth that Bringeth Destruction - Moses' Death Irrevocably Doomed - Moses Prayer for Suspension of Judgment - God Tries to Comfort Moses Concerning His Death - The Intercessions for Moses - Moses Serves Joshua Chapter VII The Last Day of Moses' Life - Moses Beholds the Future - Moses Meets the Messiah in Heaven - The Last Hours of Moses The Blessing of Moses - Moses Prays for Death - Samuel Chastised by Moses - God Kisses Moses' Soul - The Mourning for Moses - Samuel's Vain Search - Moses Excels All Pious Men Chapter I: Joshua The Servant of Moses - Entering the Promised Land - Conquest of the Land - The Sun Obeys Joshua - War with the Armenians - Allotment of the Land."
    spanCorruption = generator.t5_single_example_denoise(features,
                                              noise_mask_fn=generator.random_spans_noise_mask,
                                              inputs_fn=generator.noise_span_to_unique_sentinel,
                                              targets_fn=generator.nonnoise_span_to_unique_sentinel)
    print('src: %s' % (generator.tokenizer.detokenize(spanCorruption['src'])))
    print('tgt: %s' % (generator.tokenizer.detokenize(spanCorruption['tgt'])))


    with open('data/WikiMatrix.en-ko.en', 'r') as inF:
        for line in inF:
            features = line.strip()
            spanCorruption = generator.t5_single_example_denoise(features,
                                                      noise_mask_fn=generator.random_spans_noise_mask,
                                                      inputs_fn=generator.noise_span_to_unique_sentinel,
                                                      targets_fn=generator.nonnoise_span_to_unique_sentinel)
            print('src: %s' % (generator.tokenizer.detokenize(spanCorruption['src'])))
            print('tgt: %s' % (generator.tokenizer.detokenize(spanCorruption['tgt'])))




if __name__ == "__main__":
    for t in _sentinelTokens:
        print(t)
    run_main()